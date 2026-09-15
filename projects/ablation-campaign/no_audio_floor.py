#!/usr/bin/env python3
"""No-audio floor: text-only NLL of the frozen MA backbone.

Week 1 Track A, `plan/01-interface-recipe.md` section 1 ("Two controls settle
the diagnosis without training ... 1. The no-audio floor."). Scores the
frozen text decoder named in `model.decoder.name` -- no encoder, no adapter,
no audio at all -- on the exact per-language validation cuts an MA arm's
`eval_loss` is computed over (same seed, same `max_samples` cap, same
`text_field` resolution), rendered through the same chat-templated MA prompt
(`"{audio_token}"` as the user turn, transcript as the assistant turn, same
label masking). If this floor matches the observed MA eval loss, the adapter
never conditioned the decoder on audio.

Deliberately does not reuse `MELTMapDataset`/`MELTDataCollator`: both call
`cut.load_audio()` unconditionally, which would read the full validation
audio for a run that never uses it. `MELTForCausalLM.forward` takes
`input_features=None` cleanly (`modeling_melt.py`'s `_merge_embeddings` only
runs `if input_features is not None`), so the exact same text formatting
helpers the collator uses are called here directly, then scored without ever
touching a waveform.

Usage (inside the training container, single GPU):
    python projects/ablation-campaign/no_audio_floor.py \\
        --config projects/ablation-campaign/ABL-MA-700-asr.yaml \\
        --output /workspace/outputs/no_audio_floor/no_audio_floor.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import set_seed

# Repo root (this file lives at <root>/projects/ablation-campaign/), so `melt`
# imports whether invoked as `python path/to/this.py` (which puts only this
# file's own directory on sys.path) or from an unrelated cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from melt.logging_utils import configure_logging, get_logger  # noqa: E402
from melt.modeling import MELTForCausalLM  # noqa: E402
from melt.training.config import get_default_config  # noqa: E402
from melt.training.data.audio.lhotse import (  # noqa: E402
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.data.audio.lhotse.helpers import (  # noqa: E402
    _get_config_value,
    _normalize_prompt_template,
    apply_chat_template_to_texts,
    get_tags_from_cut,
    get_text_from_cut,
    mask_non_assistant_tokens,
)
from melt.training.data.chat_templates import (  # noqa: E402
    get_chat_template_config,
    validate_chat_template_config,
)
from melt.training.setup import prepare_melt_config, prepare_processor  # noqa: E402

logger = get_logger(__name__)


def resolve_text_field_for_cut(cut, ds_text_field: str) -> str:
    """Per-cut `tags.text_field` overrides the dataset-level default.

    Mirrors `MELTMapDataset._resolve_text_field` exactly -- ST sources tag
    their own cuts with a different field (e.g. `custom.translation_en`) and
    this has to agree or a cut passes with one field and scores with another.
    """
    if hasattr(cut, "tags") and cut.tags:
        text_field = cut.tags.get("text_field")
        if text_field:
            return str(text_field)
    return ds_text_field


def build_items(cuts, ds_text_field: str, strict: bool, skip_on_mismatch: bool):
    """Extract (text, task, lang, ...) from cuts without loading any audio.

    Mirrors `MELTMapDataset.__getitem__`'s text/tag extraction, including the
    lowercasing -- the MA arms trained on lowercased transcripts, and this
    floor has to render the same input.
    """
    items = []
    skipped = 0
    for cut in cuts:
        text_field = resolve_text_field_for_cut(cut, ds_text_field)
        text = get_text_from_cut(cut, text_field, strict=strict, skip_on_mismatch=skip_on_mismatch)
        if not text or not text.strip():
            skipped += 1
            continue
        text = text.strip().lower()
        task, lang, src_lang, tgt_lang = get_tags_from_cut(cut)
        items.append(
            {
                "text": text,
                "task": task,
                "lang": lang,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "cut_id": cut.id,
            }
        )
    return items, skipped


def score_named_set(name, sub_cfg, processor, model, assistant_start_ids, assistant_end_ids,
                     prompt_template, prompt_template_selection, strict_text_field,
                     skip_on_mismatch, batch_size, device):
    logger.info("Materialising cuts for %s (metadata only, no audio read)...", name)
    cuts = materialize_cuts_for_eval(sub_cfg)
    ds_text_field = str(_get_config_value(sub_cfg, "text_field", "text"))
    items, skipped = build_items(cuts, ds_text_field, strict_text_field, skip_on_mismatch)
    logger.info("%s: %d valid cuts (%d skipped of %d)", name, len(items), skipped, len(cuts))

    nll_sum = 0.0
    token_count = 0

    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        texts = [it["text"] for it in batch_items]
        tasks = [it["task"] for it in batch_items]
        langs = [it["lang"] for it in batch_items]
        src_langs = [it["src_lang"] for it in batch_items]
        tgt_langs = [it["tgt_lang"] for it in batch_items]

        formatted = apply_chat_template_to_texts(
            texts,
            tasks,
            langs,
            tokenizer=processor.tokenizer,
            audio_token=processor.audio_token,
            prompt_template=prompt_template,
            prompt_template_selection=prompt_template_selection,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            return_prompts=False,
        )

        # audio=None: MELTProcessor.__call__ still applies
        # `_surround_bos_eos_mm_tokens` and tokenizes exactly as it would with
        # real audio -- only `_process_audio` is skipped -- so `input_ids` are
        # byte-identical to what the real collator would have produced.
        batch = processor(text=formatted, audio=None, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels = mask_non_assistant_tokens(labels, assistant_start_ids, assistant_end_ids)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    input_features=None,
                )

        # The loss function shifts internally (pads labels by one -100 then
        # slices), so `labels[..., 1:]` always has the same valid-token count
        # as the tensor it actually scores against -- the trailing dropped
        # position and the appended pad position are both -100.
        valid = int((labels[..., 1:] != -100).sum().item())
        nll_sum += float(out.loss.item()) * valid
        token_count += valid

    nats_per_token = nll_sum / max(token_count, 1)
    logger.info("%s: %.4f nats/token over %d target tokens (%d cuts)", name, nats_per_token, token_count, len(items))
    return {
        "nats_per_token": nats_per_token,
        "num_cuts": len(items),
        "num_target_tokens": token_count,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="MA config, e.g. ABL-MA-700-asr.yaml")
    ap.add_argument("--output", required=True, help="Path to write the results JSON")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    configure_logging()

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))

    # Same seed the real MA-700-llama1b-ins arm trains with (campaign.yaml),
    # consumed in the same order (before model construction) so the randomly
    # initialised rows `resize_token_embeddings` adds for the audio special
    # tokens match what that arm's frozen decoder actually carries.
    seed = int(cfg.trainer.get("seed", 42))
    set_seed(seed)

    processor = prepare_processor(cfg)
    melt_config = prepare_melt_config(cfg, processor)

    logger.info("Building MELTForCausalLM (decoder=%s)...", cfg.model.decoder.name)
    model = MELTForCausalLM(melt_config, load_backbones=True)

    # Same post-construction fixups train.py's prepare_model() applies.
    if len(processor.tokenizer) > melt_config.vocab_size:
        model.text_decoder.resize_token_embeddings(
            len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8
        )
    pad_token_id = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.pad_token])[0]
    model.text_decoder.config.pad_token_id = pad_token_id

    model.to(args.device)
    model.eval()

    eval_data_config = resolve_eval_data_config(cfg.data)
    if eval_data_config is None:
        raise SystemExit(f"{args.config} has no data.validation_ds")

    apply_chat_template = bool(_get_config_value(eval_data_config, "apply_chat_template", False))
    if not apply_chat_template:
        raise SystemExit(
            "no_audio_floor expects an MA-style config with apply_chat_template: true"
        )

    ct_name = str(_get_config_value(eval_data_config, "chat_template_config", "chatml"))
    ct_cfg = get_chat_template_config(ct_name)
    validate_chat_template_config(processor.tokenizer, ct_cfg, ct_name)
    assistant_start_ids = processor.tokenizer.encode(ct_cfg.assistant_start, add_special_tokens=False)
    assistant_end_ids = processor.tokenizer.encode(ct_cfg.assistant_end, add_special_tokens=False)

    raw_prompt_template = _get_config_value(eval_data_config, "prompt_template", None)
    prompt_template = _normalize_prompt_template(raw_prompt_template)
    prompt_template_selection = str(
        _get_config_value(eval_data_config, "prompt_template_selection", "random")
    )
    strict_text_field = bool(_get_config_value(eval_data_config, "strict_text_field", False))
    skip_on_mismatch = bool(_get_config_value(eval_data_config, "skip_text_field_mismatch", False))

    named = split_eval_config_by_name(eval_data_config)
    if named is None:
        named = {"eval": eval_data_config}

    results = {}
    grand_nll_sum = 0.0
    grand_token_count = 0
    for name, sub_cfg in named.items():
        r = score_named_set(
            name,
            sub_cfg,
            processor,
            model,
            assistant_start_ids,
            assistant_end_ids,
            prompt_template,
            prompt_template_selection,
            strict_text_field,
            skip_on_mismatch,
            args.batch_size,
            args.device,
        )
        results[name] = r
        grand_nll_sum += r["nats_per_token"] * r["num_target_tokens"]
        grand_token_count += r["num_target_tokens"]

    results["_overall"] = {
        "nats_per_token": grand_nll_sum / max(grand_token_count, 1),
        "num_target_tokens": grand_token_count,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "decoder": str(cfg.model.decoder.name),
                "seed": seed,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
