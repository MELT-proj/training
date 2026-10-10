"""Text-only ceiling of a speech-translation frozen set: the frozen decoder translating the gold transcript.

For every sample of melt-eval ST frozen sets (their ``manifest.jsonl``), the
speech LLM's own text decoder gets the mirror teacher's prompt -- the student's
translate instruction with the source-language transcript (``source_text``,
cased and punctuated) where the audio would be, rendered by
``build_distill_prompts`` -- and decodes greedily under the trainer's stop and
vocabulary rules. chrF and BLEU per direction against the set's reference are
the most a speech model distilled from this teacher could reach on that set,
from the content side. Samples without ``source_text`` (FLEURS ga) are skipped.

Two steps, because the training image has no sacrebleu:

    # inside the training image (projects/self-distill/launch_script.sh); the frozen-set
    # directories must be visible there, e.g. copied under the outputs
    python projects/self-distill/text_ceiling.py generate --ckpt /workspace/outputs/<GOLDW run> \\
        --sets /workspace/outputs/sd-text-ceiling/sets/* --out /workspace/outputs/sd-text-ceiling/dev.jsonl
    # anywhere sacrebleu imports (the melt-eval venv); scored as summarize_evals.py scores the speech runs
    python projects/self-distill/text_ceiling.py score .../dev.jsonl --json .../dev.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


REPEAT = "Repeat the following content exactly, word for word, and write nothing else.\n\n{audio_token}"
TRANSLATE = "Translate the following content into {tgt_lang}, and write nothing else.\n\n{audio_token}"


def read_rows(set_dir: Path) -> list[dict]:
    with open(set_dir / "manifest.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r.get("task") == "st" and (r.get("source_text") or "").strip()]


def translate(model, processor, rows: list[dict], template: str, batch_size: int, max_new_tokens: int) -> list[str]:
    """Greedy teacher replies for *rows*, length-sorted batches, as the trainer's teacher rollout decodes."""
    import torch

    from melt.training.self_distill import build_distill_prompts, distill_vocab_limit

    tokenizer = processor.tokenizer
    eos = model.text_decoder.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    eos_ids = sorted({int(e) for e in eos if e is not None})
    vocab_limit = distill_vocab_limit(processor)
    logits_dim = model.text_decoder.get_output_embeddings().weight.shape[0]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["source_text"]))
    out: list[str] = [""] * len(rows)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = [rows[i] for i in idx]
        prompts = build_distill_prompts(
            processor,
            [r["source_text"].strip() for r in batch],
            ["st"] * len(batch),
            [r["src_lang"] for r in batch],
            {"asr": REPEAT, "st": template},
            teacher_prompt="mirror",
            tgt_langs=[r["tgt_lang"] for r in batch],
        )
        ids = prompts["teacher_prompt_input_ids"].cuda()
        with torch.inference_mode():
            gen = model.text_decoder.generate(
                input_ids=ids,
                attention_mask=prompts["teacher_prompt_attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                suppress_tokens=list(range(vocab_limit, logits_dim)),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )[:, ids.shape[1] :].tolist()
        for row, i in zip(gen, idx):
            stop = next((k for k, t in enumerate(row) if t in eos_ids), None)
            out[i] = tokenizer.decode(row if stop is None else row[:stop], skip_special_tokens=True).strip()
    return out


def generate(args: argparse.Namespace) -> None:
    from alignment_probe import load_checkpoint
    from omegaconf import OmegaConf

    ckpt = Path(args.ckpt)
    model, processor = load_checkpoint(ckpt, OmegaConf.load(ckpt / "resolved_config.json"))
    template = args.template.replace("\\n", "\n")
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        for set_dir in map(Path, args.sets):
            rows = read_rows(set_dir)
            hyps = translate(model, processor, rows, template, args.batch_size, args.max_new_tokens)
            for r, h in zip(rows, hyps):
                record = {"set": set_dir.name, "sample_key": r["sample_key"], "src_lang": r["src_lang"]}
                record.update(tgt_lang=r["tgt_lang"], ref=str(r["target"]).strip(), hyp=h, template=template)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[ceiling] {set_dir.name}: {len(rows)} samples, {time.time() - start:.0f} s elapsed", flush=True)


def score(args: argparse.Namespace) -> None:
    import sacrebleu

    by_direction = defaultdict(lambda: ([], []))
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps, refs = by_direction[(r["set"], r["src_lang"], r["tgt_lang"])]
            hyps.append(r["hyp"])
            refs.append(r["ref"])
    report = defaultdict(dict)
    for (name, src, tgt), (h, r) in sorted(by_direction.items()):
        report[name][src] = {
            "n": len(h),
            "chrf": sacrebleu.corpus_chrf(h, [r]).score,
            "bleu": sacrebleu.corpus_bleu(h, [r]).score,
        }
        print(
            f"{name} {src}->{tgt}: n {len(h)} chrF {report[name][src]['chrf']:.1f} BLEU {report[name][src]['bleu']:.1f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="greedy text-only translations, one JSON line per sample")
    gen.add_argument("--ckpt", required=True, help="a finished run's root (weights, processor, resolved_config.json)")
    gen.add_argument("--sets", nargs="+", required=True, help="melt-eval frozen-set directories (manifest.jsonl)")
    gen.add_argument("--out", required=True, help="samples JSONL")
    gen.add_argument(
        "--template", default=TRANSLATE, help="the student's translate instruction; literal \\n is a newline"
    )
    gen.add_argument("--batch-size", type=int, default=64)
    gen.add_argument("--max-new-tokens", type=int, default=448)
    gen.set_defaults(func=generate)
    sc = sub.add_parser("score", help="chrF and BLEU per (set, direction), as summarize_evals.py scores ST")
    sc.add_argument("samples")
    sc.add_argument("--json", default=None)
    sc.set_defaults(func=score)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
