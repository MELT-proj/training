"""Content-level alignment probe for a MELT checkpoint (self-distillation, phase 2).

Two measurements on a run's own in-training eval subset, both independent of
the objective the checkpoint was trained with:

* **retrieval**: per language, mean-pool the frozen decoder's hidden states
  over the audio frames (audio path, the run's training prompt) and over the
  transcript tokens (text path, the transcript as the teacher reads it), centre
  each modality, and rank every utterance's transcript among the language's
  candidates by cosine similarity. R@1 and MRR against chance; also within
  corpus (CV22 / MLS / VoxPopuli), which removes register as a cue. Language
  identity and register alone score at chance, so this separates content
  alignment from the coarse plateau phase 1's gap metric could not tell apart.
* **divergence**: per-token JSD, KL(text || audio) and KL(audio || text)
  between the two paths' next-token distributions, teacher-forced on the gold
  transcript as the reply. Neither path produced that reply, so the numbers
  favour neither a forward- nor a reverse-KL objective.

The subset, the prompts and the logits come from the trainer's own code
(``materialize_cuts_for_eval``, ``SelfDistillEvalCollator``, ``response_logits``),
and audio positions in the merged sequence from the model's own ``_inject_tensor``.

Usage (training image, 1 GPU, dataset root bound at the path the run saw):
    python projects/self-distill/alignment_probe.py --ckpt /workspace/outputs/<run>/checkpoint-1200 \\
        --out /workspace/outputs/<run>/probe-1200.json
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from melt.modeling import MELTConfig, MELTForCausalLM, MELTProcessor
from melt.training.data.audio.lhotse import (
    MELTMapDataset,
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.self_distill import SelfDistillEvalCollator, distill_vocab_limit, response_logits


def load_checkpoint(ckpt: Path, run_cfg, processor_dir: Path | None = None) -> tuple[MELTForCausalLM, MELTProcessor]:
    """As train.py's model.ckpt branch: the attention implementations come from the run config.

    ``checkpoint-N`` directories carry no processor; pass the one
    save_run_processor.py wrote for the run.
    """
    config = MELTConfig.from_pretrained(ckpt)
    decoder_attn = OmegaConf.select(run_cfg, "model.decoder.attn_implementation")
    encoder_attn = OmegaConf.select(run_cfg, "model.encoder.attn_implementation")
    if decoder_attn:
        config.text_decoder_config._attn_implementation = decoder_attn
    if encoder_attn:
        config.audio_encoder_config._attn_implementation = encoder_attn
    processor = MELTProcessor.from_pretrained(processor_dir or ckpt)
    model = MELTForCausalLM.from_pretrained(ckpt, config=config, dtype=torch.bfloat16)
    return model.to("cuda").eval(), processor


def corpus_of(sub, max_ids: set[str]) -> dict[str, str]:
    """cut id -> corpus (first path component under the dataset root), for the selected ids."""
    owner = {}
    for src in sub.input_cfg:
        one = copy.deepcopy(sub)
        one.input_cfg = [src]
        one.max_samples = None
        corpus = Path(str(src.shar_path)).parts[-3] if len(Path(str(src.shar_path)).parts) >= 3 else str(src.shar_path)
        for cut in materialize_cuts_for_eval(one):
            if cut.id in max_ids:
                owner[cut.id] = corpus
    return owner


@torch.no_grad()
def audio_vectors(model, batch, layers: list[int]) -> list[torch.Tensor] | None:
    """Mean hidden state over the real audio frames of the student prompt, per layer."""
    ids = batch["student_prompt_input_ids"].cuda()
    mask = batch["student_prompt_attention_mask"].cuda()
    feats = batch["input_features"].cuda().to(model.dtype)
    fmask = batch["features_attention_mask"].cuda()
    _, enc_mask, audio_lengths, _ = model._get_audio_embeddings(
        input_features=feats, features_attention_mask=fmask, output_attentions=False,
        output_hidden_states=False, return_dict=True,
    )
    marker = model._inject_tensor(
        source_tensor=enc_mask, target_tensor=torch.zeros_like(mask), inject_token_id=model.config.audio_token_id,
        input_ids=ids, source_lengths=audio_lengths,
    )
    out = model(
        input_ids=ids, attention_mask=mask, input_features=feats, features_attention_mask=fmask,
        output_hidden_states=True, return_dict=True, logits_to_keep=1,
    )
    if out.hidden_states[0].shape[1] != marker.shape[1]:
        raise RuntimeError("merged sequence and audio marker disagree in length")
    pos = marker[0].bool()
    if not pos.any():
        return None
    return [out.hidden_states[layer][0, pos].float().mean(0) for layer in layers]


@torch.no_grad()
def text_vectors(model, tokenizer, batch, text: str, layers: list[int]) -> list[torch.Tensor] | None:
    """Mean hidden state over the transcript tokens of the teacher prompt, per layer."""
    ids = batch["teacher_prompt_input_ids"]
    mask = batch["teacher_prompt_attention_mask"]
    real = ids[0][mask[0].bool()].tolist()
    prompt = tokenizer.decode(real)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    start = prompt.rfind(text)
    if start < 0 or enc["input_ids"] != real:
        return None
    end = start + len(text)
    span = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    offset = ids.shape[1] - len(real)  # left padding (none at batch size 1)
    out = model.text_decoder(
        input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True, return_dict=True,
        logits_to_keep=1,
    )
    positions = torch.tensor([offset + k for k in span], device="cuda")
    return [out.hidden_states[layer][0, positions].float().mean(0) for layer in layers]


@torch.no_grad()
def divergences(model, tokenizer, batch, text: str, vocab_limit: int) -> dict[str, float]:
    """Per-token JSD and both KLs between the paths, teacher-forced on the gold transcript."""
    response = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    ones = torch.ones_like(response)
    audio = response_logits(
        model, batch["student_prompt_input_ids"].cuda(), batch["student_prompt_attention_mask"].cuda(),
        response, ones, vocab_limit, input_features=batch["input_features"].cuda().to(model.dtype),
        features_attention_mask=batch["features_attention_mask"].cuda(),
    )
    textp = response_logits(
        model.text_decoder, batch["teacher_prompt_input_ids"].cuda(), batch["teacher_prompt_attention_mask"].cuda(),
        response, ones, vocab_limit,
    )
    p = F.log_softmax(audio[0].float(), -1)  # audio path
    q = F.log_softmax(textp[0].float(), -1)  # text path
    m = torch.logsumexp(torch.stack([p, q]), 0) - math.log(2)
    jsd = 0.5 * (p.exp() * (p - m)).sum(-1) + 0.5 * (q.exp() * (q - m)).sum(-1)
    return {
        "tokens": response.shape[1],
        "jsd": jsd.sum().item(),
        "kl_text_audio": (q.exp() * (q - p)).sum(-1).sum().item(),
        "kl_audio_text": (p.exp() * (p - q)).sum(-1).sum().item(),
        "nll_audio": -p.gather(-1, response[0].unsqueeze(-1)).sum().item(),
        "nll_text": -q.gather(-1, response[0].unsqueeze(-1)).sum().item(),
    }


def retrieval(audio: torch.Tensor, text: torch.Tensor, groups: list[str] | None = None) -> dict[str, float]:
    """R@1 / MRR of audio->text and text->audio after centring each modality (cosine)."""
    a = F.normalize(audio - audio.mean(0), dim=-1)
    t = F.normalize(text - text.mean(0), dim=-1)
    sim = a @ t.T
    if groups is not None:
        same = torch.tensor([[gi == gj for gj in groups] for gi in groups], device=sim.device)
        sim = sim.masked_fill(~same, float("-inf"))
        candidates = same.sum(1).float()
    else:
        candidates = torch.full((sim.shape[0],), float(sim.shape[0]), device=sim.device)
    out = {"chance_r1": (1 / candidates).mean().item()}
    for name, s in (("a2t", sim), ("t2a", sim.T)):
        rank = (s > s.diag()[:, None]).sum(1).float()
        out[f"r1_{name}"] = (rank == 0).float().mean().item()
        out[f"mrr_{name}"] = (1 / (rank + 1)).mean().item()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint directory (a run root or a checkpoint-N)")
    p.add_argument("--run-config", default=None, help="default: resolved_config.json in the ckpt dir or its parent")
    p.add_argument("--processor", default=None, help="processor dir (save_run_processor.py) for checkpoint-N dirs")
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=100, help="utterances per named eval set")
    p.add_argument("--prompt-template", default=None, help="student template; default: the run's data.prompt_template")
    p.add_argument("--teacher-prompt", default="bare", choices=["bare", "mirror"])
    p.add_argument("--layers", default=None, help="comma-separated hidden-state indices; default 0 and quartiles")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    run_config = args.run_config or next(
        (str(c) for c in (ckpt / "resolved_config.json", ckpt.parent / "resolved_config.json") if c.exists()), None
    )
    if run_config is None:
        raise SystemExit("no resolved_config.json next to the checkpoint; pass --run-config")
    cfg = OmegaConf.load(run_config)
    cfg.data.validation_ds.max_samples = args.max_samples
    model, processor = load_checkpoint(ckpt, cfg, Path(args.processor) if args.processor else None)
    tokenizer = processor.tokenizer
    n_layers = model.text_decoder.config.get_text_config().num_hidden_layers
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(
        {0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers}
    )
    template = args.prompt_template or OmegaConf.select(cfg, "data.prompt_template")
    eval_cfg = resolve_eval_data_config(cfg.data)
    named = split_eval_config_by_name(eval_cfg) or {"all": eval_cfg}
    collator = SelfDistillEvalCollator(
        processor=processor, config=eval_cfg, student_prompt_template=template, teacher_prompt=args.teacher_prompt
    )
    vocab_limit = distill_vocab_limit(processor)
    print(f"[probe] {ckpt} | layers {layers} | template {template!r} | teacher {args.teacher_prompt}", flush=True)

    report = {"ckpt": str(ckpt), "run_config": run_config, "layers": layers, "template": template,
              "teacher_prompt": args.teacher_prompt, "sets": {}}
    start = time.time()
    for name, sub in named.items():
        cuts = materialize_cuts_for_eval(sub)
        owner = corpus_of(sub, {c.id for c in cuts})
        dataset = MELTMapDataset(cuts=cuts, processor=processor, config=sub, is_train=False, return_langs=True)
        audio_rows, text_rows, groups = [], [], []
        div = {"tokens": 0, "jsd": 0.0, "kl_text_audio": 0.0, "kl_audio_text": 0.0, "nll_audio": 0.0, "nll_text": 0.0}
        skipped = 0
        for i in range(len(dataset)):
            item = dataset[i]
            if item.get("__invalid__"):
                skipped += 1
                continue
            batch = collator([item])
            a = audio_vectors(model, batch, layers)
            t = text_vectors(model, tokenizer, batch, item["text"], layers)
            if a is None or t is None:
                skipped += 1
                continue
            audio_rows.append(torch.stack(a))
            text_rows.append(torch.stack(t))
            groups.append(owner.get(dataset.cuts[dataset._valid_indices[i]].id, "?"))
            for k, v in divergences(model, tokenizer, batch, item["text"], vocab_limit).items():
                div[k] += v
        audio_m = torch.stack(audio_rows)  # (N, layers, d)
        text_m = torch.stack(text_rows)
        per_layer = {}
        for j, layer in enumerate(layers):
            per_layer[layer] = {
                "all": retrieval(audio_m[:, j], text_m[:, j]),
                "within_corpus": retrieval(audio_m[:, j], text_m[:, j], groups),
            }
        n_tok = max(1, div.pop("tokens"))
        report["sets"][name] = {
            "n": len(audio_rows), "skipped": skipped, "corpora": {g: groups.count(g) for g in sorted(set(groups))},
            "per_token": {k: v / n_tok for k, v in div.items()}, "retrieval": per_layer,
        }
        best = max(per_layer, key=lambda layer: per_layer[layer]["within_corpus"]["r1_a2t"])
        print(
            f"[probe] {name}: n={len(audio_rows)} skipped={skipped} | JSD/token {div['jsd'] / n_tok:.3f} "
            f"| R@1 a->t within corpus {per_layer[best]['within_corpus']['r1_a2t']:.2f} at layer {best} "
            f"(chance {per_layer[best]['within_corpus']['chance_r1']:.3f}) | {time.time() - start:.0f} s",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
