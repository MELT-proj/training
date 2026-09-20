#!/usr/bin/env python3
"""Step 0b pre-flight 1b: full-set decoding diagnostic.

plan/01-interface-recipe.md §2b, pre-flight step 1b (Fondue Orchestrator's
resolution of the "insertions dominate" STOP, main commit ad07b3f): the 20
sample pairs the trainer logs are not representative (WER 1.19 on them
against the trainer's own 0.62/0.78 over the full 200-utterance dev sets),
so before trusting a runaway-fraction number, measure it on every hypothesis
in the full dev-clean/dev-other sets, greedy exactly as training decodes,
and correlate runaway against reference duration (the "losing its place in
long 50 Hz audio" hypothesis R-k5 will test). A second pass with
`no_repeat_ngram_size=4` sizes how much of the WER the loops explain.
Diagnostic only -- does not gate step 0b's arms.

Loads the trained checkpoint the same way `train.py`'s `prepare_model` does
for `model.ckpt` (MELTConfig/MELTProcessor/MELTForCausalLM.from_pretrained,
with the attn_implementation fixup a checkpoint's config.json never
records), then builds the real eval pipeline (materialize_cuts_for_eval ->
MELTMapDataset -> MELTDataCollator, `max_samples` removed) and calls
`model.generate()` directly -- deliberately not through MELTTrainer, which
wires in DDP/FSDP unsharding and W&B this single-GPU diagnostic does not
need.

Usage (inside the training container, single GPU):
    python projects/ablation-campaign/step0b_diagnostic_full_eval.py \\
        --config projects/ablation-campaign/ABL-MA-librispeech.yaml \\
        --checkpoint /workspace/outputs/MA-librispeech-w2vbF-llama1bInsF-mlpT-ga1-elr6e6-dlr2e5-lr1e3-s44-8g \\
        --output-dir /workspace/outputs/step0b_diagnostic
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import jiwer
import torch
from omegaconf import OmegaConf
from transformers import set_seed

# Repo root (this file lives at <root>/projects/ablation-campaign/), so `melt`
# imports whether invoked as `python path/to/this.py` (which puts only this
# file's own directory on sys.path) or from an unrelated cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from melt.evaluation import BasicTextNormalizer  # noqa: E402
from melt.logging_utils import configure_logging, get_logger  # noqa: E402
from melt.modeling import MELTConfig, MELTForCausalLM, MELTProcessor  # noqa: E402
from melt.training.config import get_default_config  # noqa: E402
from melt.training.data.audio.lhotse import (  # noqa: E402
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.data.audio.lhotse.collator import MELTDataCollator  # noqa: E402
from melt.training.data.audio.lhotse.helpers import _get_config_value  # noqa: E402
from melt.training.data.audio.lhotse.map_dataset import MELTMapDataset  # noqa: E402
from melt.training.metrics import _mean_length_ratio, _runaway_fraction  # noqa: E402

logger = get_logger(__name__)


def load_model_from_checkpoint(ckpt_dir: Path, decoder_attn_implementation: str | None,
                                encoder_attn_implementation: str | None, device: str):
    """Mirrors train.py's prepare_model() ckpt branch exactly (see that
    function's comment for why attn_implementation needs the fixup: a
    checkpoint's config.json never records it, and loading one without this
    silently falls back to sdpa-on-cuDNN, ~12x slower for generation)."""
    config = MELTConfig.from_pretrained(ckpt_dir)
    if decoder_attn_implementation:
        config.text_decoder_config._attn_implementation = decoder_attn_implementation
    if encoder_attn_implementation:
        config.audio_encoder_config._attn_implementation = encoder_attn_implementation

    processor = MELTProcessor.from_pretrained(ckpt_dir)
    model = MELTForCausalLM.from_pretrained(ckpt_dir, config=config)
    model.to(device)
    model.eval()
    return model, processor


def build_named_loaders(cfg, processor, batch_size: int):
    """One (cut_id -> duration, DataLoader) pair per named validation set,
    with `max_samples` removed so the full set is scored."""
    eval_data_config = resolve_eval_data_config(cfg.data)
    named = split_eval_config_by_name(eval_data_config)
    if named is None:
        named = {"eval": eval_data_config}

    loaders = {}
    for name, sub_cfg in named.items():
        sub_cfg = copy.deepcopy(sub_cfg)
        OmegaConf.set_struct(sub_cfg, False)
        sub_cfg.max_samples = None

        cuts = materialize_cuts_for_eval(sub_cfg)
        durations = {cut.id: float(cut.duration) for cut in cuts}

        dataset = MELTMapDataset(cuts=cuts, processor=processor, config=sub_cfg, is_train=False)
        collator = MELTDataCollator(processor=processor, config=sub_cfg, is_train=False)

        # MELTDataCollator drops cut_id from the batch it returns (it only
        # keeps what the model/loss need); wrap it to also emit the id list
        # in the same order, filtered the same way (`__invalid__` items
        # dropped first, identical to the real collator's own filtering).
        def collate_with_ids(items, _collator=collator):
            valid_ids = [it["cut_id"] for it in items if not it.get("__invalid__", False)]
            batch = _collator(items)
            batch["cut_ids"] = valid_ids
            return batch

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_with_ids,
            shuffle=False, drop_last=False, num_workers=2,
        )
        loaders[name] = (durations, loader)
        logger.info("%s: %d valid cuts (full set, max_samples removed)", name, len(dataset))
    return loaders


def run_pass(name, durations, loader, model, processor, normalizer, device,
             max_new_tokens, num_beams, no_repeat_ngram_size, out_path):
    gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=num_beams, use_cache=True)
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    refs_raw, hyps_raw, cut_ids_all = [], [], []
    with open(out_path, "w") as f:
        for batch in loader:
            cut_ids = batch.pop("cut_ids")
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            input_features = batch.get("input_features")
            if input_features is not None:
                input_features = input_features.to(device=device, dtype=torch.bfloat16)
            features_attention_mask = batch.get("features_attention_mask")
            if features_attention_mask is not None:
                features_attention_mask = features_attention_mask.to(device)

            # References: the label span, same pipeline as training (chat
            # scaffolding included), decoded and normalized just like the
            # hypotheses so the two sides go through the identical text
            # pipeline TrainingEvaluator uses -- not the pre-template `text`
            # field, which would skip whatever the tokenizer/template does.
            labels = batch["labels"]

            with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    input_features=input_features,
                    features_attention_mask=features_attention_mask,
                    **gen_kwargs,
                )

            pad_id = processor.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = processor.tokenizer.eos_token_id
            labels_for_decode = torch.where(labels == -100, torch.full_like(labels, pad_id), labels)

            batch_refs = processor.batch_decode(labels_for_decode, skip_special_tokens=True)
            batch_hyps = processor.batch_decode(generated, skip_special_tokens=True)

            for cut_id, ref, hyp in zip(cut_ids, batch_refs, batch_hyps):
                ref_n = normalizer(ref).strip()
                hyp_n = normalizer(hyp).strip()
                refs_raw.append(ref_n)
                hyps_raw.append(hyp_n)
                cut_ids_all.append(cut_id)
                f.write(json.dumps({
                    "cut_id": cut_id,
                    "duration": durations.get(cut_id),
                    "reference": ref_n,
                    "hypothesis": hyp_n,
                    "ref_words": len(ref_n.split()),
                    "hyp_words": len(hyp_n.split()),
                }) + "\n")

            logger.info("%s: %d/%d cuts decoded", name, len(refs_raw), len(loader.dataset))

    measures = jiwer.process_words(refs_raw, hyps_raw)
    n_ref_words = sum(len(r.split()) for r in refs_raw)
    length_ratio = _mean_length_ratio(refs_raw, hyps_raw)
    runaway_fraction = _runaway_fraction(refs_raw, hyps_raw)

    runaway_durations, non_runaway_durations = [], []
    for cut_id, ref, hyp in zip(cut_ids_all, refs_raw, hyps_raw):
        rw = len(ref.split())
        if rw == 0:
            continue
        d = durations.get(cut_id)
        if d is None:
            continue
        (runaway_durations if len(hyp.split()) > 2 * rw else non_runaway_durations).append(d)

    return {
        "n": len(refs_raw),
        "wer": measures.wer,
        "substitution_rate": measures.substitutions / n_ref_words,
        "deletion_rate": measures.deletions / n_ref_words,
        "insertion_rate": measures.insertions / n_ref_words,
        "length_ratio": length_ratio,
        "runaway_fraction": runaway_fraction,
        "n_runaway": len(runaway_durations),
        "mean_duration_runaway": sum(runaway_durations) / len(runaway_durations) if runaway_durations else None,
        "mean_duration_non_runaway": sum(non_runaway_durations) / len(non_runaway_durations) if non_runaway_durations else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="MA-librispeech-l4-ep3's final checkpoint dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    configure_logging()
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))
    set_seed(int(cfg.trainer.get("seed", 42)))

    decoder_attn = cfg.model.decoder.get("attn_implementation", None)
    encoder_attn = cfg.model.encoder.get("attn_implementation", None)
    model, processor = load_model_from_checkpoint(
        Path(args.checkpoint), decoder_attn, encoder_attn, args.device
    )

    loaders = build_named_loaders(cfg, processor, args.batch_size)
    normalizer = BasicTextNormalizer()

    max_new_tokens = int(cfg.trainer.get("generation_max_length", 256))
    num_beams = int(cfg.trainer.get("generation_num_beams", 1))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ngram in (None, 4):
        pass_name = "greedy" if ngram is None else f"no_repeat_ngram_size_{ngram}"
        pass_results = {}
        for name, (durations, loader) in loaders.items():
            logger.info("=== %s / %s ===", pass_name, name)
            hyp_path = out_dir / f"hypotheses_{pass_name}_{name}.jsonl"
            pass_results[name] = run_pass(
                name, durations, loader, model, processor, normalizer, args.device,
                max_new_tokens, num_beams, ngram, hyp_path,
            )
            logger.info("%s / %s: %s", pass_name, name, json.dumps(pass_results[name], indent=2))
        all_results[pass_name] = pass_results

    results_path = out_dir / "step0b_diagnostic_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {"config": args.config, "checkpoint": args.checkpoint, "results": all_results},
            f, indent=2,
        )
    logger.info("Wrote %s", results_path)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
