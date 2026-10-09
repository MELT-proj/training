"""Text-only audit of the self-distillation teacher (phase 2, step 1b).

The teacher in ``melt.training.self_distill`` is the frozen decoder reading the
transcript. Before an audio student is distilled from it, this measures what it
actually writes for each prompt the method could use, per language, with no
audio and no adapter:

* ``bare``: the transcript alone as the user turn, i.e. the teacher prompt all
  phase-1 variants trained on. Reports reply length, the share of replies
  longer than the phase-1 budget of 128 tokens (those targets never contained
  the stop token), whether the reply restates the transcript (AZeroS's
  self-elicit condition) and how often it switches to English.
* ``repeat`` / ``repeat_strict``: the transcript followed by a repeat
  instruction, i.e. the student's prompt with the audio replaced by its
  transcript. WER against the transcript is the ceiling of distillation for
  ASR: words the teacher rewrites become target errors.
* ``translate`` / ``translate_strict``: FLEURS, every ordered pair of the
  languages. chrF against the FLEURS reference in the target language is the
  ceiling of the speech-translation targets the teacher would supply.

Transcripts are read the way ``SpeechToTextDataset`` reads them
(``get_text_from_cut`` with the source's ``text_field``, then
``strip().lower()``), and prompts are rendered the way ``build_distill_prompts``
renders the teacher prompt (chat template, generation prompt, thinking off).
Decoding is greedy. WER uses the in-training normaliser (``BasicTextNormalizer``);
chrF is sacrebleu's default chrF2 on lowercased text.

Usage (inside the training image, repo root as cwd; see launch_teacher_audit.sh):

    python projects/self-distill/teacher_audit.py --out-dir /path/to/out
"""

import argparse
import glob
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict

import jiwer
import torch
from lhotse import load_manifest_lazy
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from melt.evaluation import BasicTextNormalizer
from melt.training.data.audio.lhotse.helpers import get_text_from_cut


LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian"}
FLEURS_LOCALES = {"en": "en_us", "de": "de_de", "fr": "fr_fr", "es": "es_419", "it": "it_it"}

# "{content}" is the transcript here; a student prompt puts "{audio_token}" in its place.
INSTRUCTIONS = {
    "bare": "{content}",
    "repeat": "{content} Repeat the above content.",
    "repeat_strict": "{content} Repeat the above content exactly, word for word, and write nothing else.",
    "translate": "{content} Translate the above content into {target}.",
    "translate_strict": "{content} Translate the above content into {target}. Write only the translation.",
}

# Phase-1 training rollouts were capped here; a longer reply lost its stop token.
PHASE1_BUDGET = 128

# Frequent English function words that are not also common words in de/fr/es/it.
_EN_FUNCTION_WORDS = frozenset(
    "the and is of to that this you for with are can your what here about which have has they there would".split()
)

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _cut_files(shar_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(shar_dir, "cuts.*.jsonl")) + glob.glob(os.path.join(shar_dir, "cuts.*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no cuts.*.jsonl[.gz] in {shar_dir}")
    return sorted(paths)


def _iter_cuts(shar_dir: str):
    for path in _cut_files(shar_dir):
        yield from load_manifest_lazy(path)


def read_validation_texts(config_path: str, langs: list[str], per_source: int) -> list[dict]:
    """The first ``per_source`` usable utterances of each validation source, as the dataset reads them."""
    cfg = OmegaConf.load(config_path)
    vds = cfg.data.validation_ds
    sources = OmegaConf.to_container(vds.input_cfg, resolve=True)
    default_field = vds.get("text_field", "text")
    min_d, max_d = float(vds.get("min_duration", 0.0)), float(vds.get("max_duration", 1e9))
    root = os.environ.get("LOCAL_DATASETS_DIR", "")
    rows = []
    for src in sources:
        tags = src.get("tags") or {}
        lang = tags.get("lang")
        if lang not in langs:
            continue
        text_field = tags.get("text_field", default_field)
        name = os.path.relpath(src["shar_path"], root) if root else src["shar_path"]
        kept = 0
        for cut in _iter_cuts(src["shar_path"]):
            if not min_d <= cut.duration <= max_d:
                continue
            try:
                text = get_text_from_cut(cut, text_field, strict=bool(vds.get("strict_text_field", False)))
            except ValueError:
                continue
            if not text or len(text.strip()) < 3:
                continue
            rows.append({"source": name, "lang": lang, "id": cut.id, "text": text.strip().lower()})
            kept += 1
            if kept >= per_source:
                break
        print(f"[audit] {name}: {kept} utterances (text_field={text_field})", flush=True)
    return rows


def read_fleurs(root: str, split: str, langs: list[str], n: int) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Sentence ids present in every language's FLEURS ``split``, and each language's PNC text by id."""
    texts = {}
    for lang in langs:
        by_id = {}
        for cut in _iter_cuts(os.path.join(root, FLEURS_LOCALES[lang], split)):
            if cut.id in by_id:  # several speakers read the same sentence
                continue
            text = get_text_from_cut(cut, "custom.pnc_text")
            if text and text.strip():
                by_id[cut.id] = text.strip()
        texts[lang] = by_id
    common = set.intersection(*(set(t) for t in texts.values()))
    ids = sorted(common, key=lambda i: (len(i), i))[:n]
    print(f"[audit] FLEURS {split}: {len(common)} sentences shared by {langs}, using {len(ids)}", flush=True)
    return ids, texts


def render(tokenizer, content: str) -> str:
    """The teacher prompt exactly as ``build_distill_prompts`` renders it."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int, eos_ids: list[int]) -> list[dict]:
    """Greedy replies, length-sorted batches, cut at the first stop token."""
    eos = set(eos_ids)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[dict | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :].tolist()
        for row, i in zip(new, idx):
            stop = next((k for k, t in enumerate(row) if t in eos), None)
            ids = row if stop is None else row[:stop]
            text = _THINK.sub("", tokenizer.decode(ids, skip_special_tokens=True)).strip()
            out[i] = {"reply": text, "n_tokens": len(ids), "hit_cap": stop is None}
    return out


def _char_ngrams(text: str, n: int) -> Counter:
    text = "".join(text.split())
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def corpus_chrf(hyps: list[str], refs: list[str], order: int = 6, beta: float = 2.0) -> float:
    """sacrebleu's default corpus chrF (char order 6, no word n-grams, beta 2, whitespace dropped)."""
    stats = [[0, 0, 0] for _ in range(order)]
    for hyp, ref in zip(hyps, refs):
        for n in range(1, order + 1):
            h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
            # sacrebleu counts no hypothesis n-grams of an order the reference is too short to have.
            stats[n - 1][0] += sum(h.values()) if r else 0
            stats[n - 1][1] += sum(r.values())
            stats[n - 1][2] += sum((h & r).values())
    prec = rec = 0.0
    effective = 0
    for n_hyp, n_ref, n_match in stats:
        if n_hyp > 0 and n_ref > 0:
            prec += n_match / n_hyp
            rec += n_match / n_ref
            effective += 1
    if effective == 0:
        return 0.0
    prec, rec = prec / effective, rec / effective
    if prec + rec == 0:
        return 0.0
    factor = beta**2
    return 100 * (1 + factor) * prec * rec / (factor * prec + rec)


def _content_recall(norm, source: str, reply: str) -> float:
    words = {w for w in norm(source).split() if len(w) >= 4}
    return len(words & set(norm(reply).split())) / len(words) if words else 0.0


def _english_like(norm, reply: str) -> bool:
    words = norm(reply).split()
    return bool(words) and sum(w in _EN_FUNCTION_WORDS for w in words) / len(words) >= 0.08


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarise(rows: list[dict], langs: list[str]) -> dict:
    norm = BasicTextNormalizer()
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    summary = {}

    for lang in langs:
        rs = [r for r in by_task.get("bare", []) if r["lang"] == lang]
        if not rs:
            continue
        tokens = [r["n_tokens"] for r in rs]
        summary.setdefault("bare", {})[lang] = {
            "n": len(rs),
            "reply_tokens_mean": _mean(tokens),
            "reply_tokens_median": statistics.median(tokens),
            f"over_{PHASE1_BUDGET}_tokens": _mean(t > PHASE1_BUDGET for t in tokens),
            "hit_cap": _mean(r["hit_cap"] for r in rs),
            "restates_transcript": _mean(norm(r["text"]) in norm(r["reply"]) for r in rs),
            "content_word_recall": _mean(_content_recall(norm, r["text"], r["reply"]) for r in rs),
            "reply_english_like": _mean(_english_like(norm, r["reply"]) for r in rs),
        }

    for task in ("repeat", "repeat_strict"):
        for lang in langs:
            rs = [r for r in by_task.get(task, []) if r["lang"] == lang and norm(r["text"]).strip()]
            if not rs:
                continue
            refs = [norm(r["text"]) for r in rs]
            hyps = [norm(r["reply"]) for r in rs]
            summary.setdefault(task, {})[lang] = {
                "n": len(rs),
                "wer": jiwer.wer(refs, hyps),
                "cer": jiwer.cer(refs, hyps),
                "exact_match": _mean(h == ref for h, ref in zip(hyps, refs)),
                "length_ratio": _mean(len(h.split()) / max(1, len(ref.split())) for h, ref in zip(hyps, refs)),
                "hit_cap": _mean(r["hit_cap"] for r in rs),
            }

    for task in ("translate", "translate_strict"):
        pairs = sorted({(r["lang"], r["target"]) for r in by_task.get(task, [])})
        for src, tgt in pairs:
            rs = [r for r in by_task[task] if r["lang"] == src and r["target"] == tgt]
            summary.setdefault(task, {})[f"{src}-{tgt}"] = {
                "n": len(rs),
                "chrf": corpus_chrf([r["reply"].lower() for r in rs], [r["reference"].lower() for r in rs]),
                "copies_source": _mean(norm(r["reply"]) == norm(r["text"]) for r in rs),
                "hit_cap": _mean(r["hit_cap"] for r in rs),
            }
    return summary


def to_markdown(summary: dict) -> str:
    lines = []
    for task, table in summary.items():
        keys = list(next(iter(table.values())).keys())
        lines += [f"### {task}", "", "| | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for name, vals in table.items():
            cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals.values()]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", default="projects/ablation-campaign/ABL-MA-700-asr.yaml")
    p.add_argument("--decoder", default="Qwen/Qwen3.5-2B")
    p.add_argument("--langs", default="en,de,fr,es,it")
    p.add_argument("--tasks", default=",".join(INSTRUCTIONS))
    p.add_argument("--per-source", type=int, default=70, help="utterances per validation source (3 sources per language)")
    p.add_argument("--fleurs-root", default=None, help="default: $LOCAL_DATASETS_DIR/fleurs")
    p.add_argument("--fleurs-split", default="validation")
    p.add_argument("--fleurs-n", type=int, default=100, help="FLEURS sentences per source language")
    p.add_argument("--bare-max-new-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--attn", default="flash_attention_2")
    args = p.parse_args()

    langs = args.langs.split(",")
    tasks = args.tasks.split(",")
    unknown = sorted(set(tasks) - set(INSTRUCTIONS))
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; choose from {sorted(INSTRUCTIONS)}")
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.decoder).get_text_config(decoder=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.decoder, config=config, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to("cuda").eval()
    # The trainer stops on the generation config's EOS ids plus MELT's eos_token for Qwen.
    eos = model.generation_config.eos_token_id
    eos_ids = sorted({*(eos if isinstance(eos, (list, tuple)) else [eos]), tokenizer.convert_tokens_to_ids("<|endoftext|>")})
    print(f"[audit] {args.decoder} ({type(model).__name__}), eos ids {eos_ids}, attn {args.attn}", flush=True)

    items = []
    if {"bare", "repeat", "repeat_strict"} & set(tasks):
        for row in read_validation_texts(args.config, langs, args.per_source):
            for task in ("bare", "repeat", "repeat_strict"):
                if task in tasks:
                    items.append({**row, "task": task, "content": INSTRUCTIONS[task].format(content=row["text"])})
    if {"translate", "translate_strict"} & set(tasks):
        root = args.fleurs_root or os.path.join(os.environ["LOCAL_DATASETS_DIR"], "fleurs")
        ids, texts = read_fleurs(root, args.fleurs_split, langs, args.fleurs_n)
        for task in ("translate", "translate_strict"):
            if task not in tasks:
                continue
            for src in langs:
                for tgt in langs:
                    if tgt == src:
                        continue
                    for sid in ids:
                        source_text = texts[src][sid].lower()  # as the dataset would feed it
                        items.append({
                            "source": f"fleurs/{FLEURS_LOCALES[src]}/{args.fleurs_split}", "lang": src, "id": sid,
                            "text": source_text, "target": tgt, "reference": texts[tgt][sid], "task": task,
                            "content": INSTRUCTIONS[task].format(content=source_text, target=LANG_NAMES[tgt]),
                        })

    start = time.time()
    for task in tasks:
        task_items = [it for it in items if it["task"] == task]
        if not task_items:
            continue
        budget = args.bare_max_new_tokens if task == "bare" else args.max_new_tokens
        replies = generate(model, tokenizer, [render(tokenizer, it["content"]) for it in task_items], budget, args.batch_size, eos_ids)
        for it, rep in zip(task_items, replies):
            it.update(rep)
        print(f"[audit] {task}: {len(task_items)} prompts, {time.time() - start:.0f} s elapsed", flush=True)

    with open(os.path.join(args.out_dir, "samples.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    summary = summarise(items, langs)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "eos_ids": eos_ids, "summary": summary}, f, indent=2)
    report = to_markdown(summary)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
