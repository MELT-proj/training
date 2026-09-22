#!/usr/bin/env python3
"""Whisper-large-v3's own WER on the step-0b dev sets (reference line for `wsd-50hz-whisper`).

plan/timeline.md, week 1 Track B, "Whisper-large-v3's own WER". Step 0b's
Whisper arm (`MA-librispeech-whisperlargeF-...-s50-8g`) scored dev-clean
0.038 / dev-other 0.061; this decodes the same two sets with Whisper itself
-- no MELT model, no adapter, no LLM decoder -- so that the arm's number can
be restated as the fraction of the encoder's own ability the projector
recovers (`03-audio-stack.md` section 0).

What is held identical to the arm, and how:

* Cuts: `ABL-MA-librispeech.yaml`'s `validation_ds`, split by name through
  `resolve_eval_data_config` / `split_eval_config_by_name` and materialised by
  `materialize_cuts_for_eval`, i.e. the same filters and the same cuts. The
  arm's `max_samples: 500` is a *seeded shuffle then truncate* inside that
  function (not the first 500 in shard order), so the 500-utterance pass here
  calls it with `max_samples=500` on the same config and gets the very same
  utterances the arm was scored on.
* Reference: `custom.pnc_text`, strict, resolved by `get_text_from_cut` with
  the same per-cut `tags.text_field` override `MELTMapDataset` honours. Cuts
  with no reference are counted and dropped exactly as `MELTMapDataset` does.
* Scoring: `BasicTextNormalizer` on hypothesis and reference, then
  `jiwer.process_words` (WER) and `jiwer.cer`, as `melt/training/metrics.py`.
* Decoding: greedy (`num_beams=1`), descending-duration batches (the order
  `MELTMapDataset` iterates in), `flash_attention_2`, bf16, language forced to
  English and task `transcribe`. The unforced pass is optional (`--unforced`).
* Cuts over 30 s: the arm's processor cuts long audio into whole 30 s windows
  and feeds all of them to the encoder (`processing_melt.py`), so the arm hears
  every cut in full. Plain Whisper short-form decoding truncates at 30 s. The
  headline therefore decodes cuts over 30 s with Whisper's own long-form
  algorithm; the truncated variant and the within-30 s-only score are reported
  beside it.

Two things it measures rather than assumes:

* Residual of the casing/punctuation/rewrite mismatch: every hypothesis is
  also scored against the untouched LibriSpeech transcript
  (`supervisions[0].text`), and the two references are scored against each
  other. The gap between the two WERs is what the PNC pass adds to (or takes
  from) the comparison after `BasicTextNormalizer`.
* Dropped cuts: raw materialised cuts, cuts with a usable reference, cuts
  decoded, and the same count with the arm's duration/token filters switched
  off, per set, so a silently shorter set cannot go unnoticed.

Run only through SLURM (`whisper_reference_decode.sbatch`).
"""

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import jiwer
import torch
from omegaconf import OmegaConf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed

# Repo root (this file lives at <root>/projects/ablation-campaign/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from melt.evaluation import BasicTextNormalizer  # noqa: E402
from melt.logging_utils import configure_logging, get_logger  # noqa: E402
from melt.training.config import get_default_config  # noqa: E402
from melt.training.data.audio.lhotse import (  # noqa: E402
    materialize_cuts_for_eval,
    resolve_eval_data_config,
    split_eval_config_by_name,
)
from melt.training.data.audio.lhotse.helpers import (  # noqa: E402
    _get_config_value,
    get_text_from_cut,
    load_audio_from_cut,
)

logger = get_logger(__name__)

SAMPLE_RATE = 16000
WHISPER_WINDOW_S = 30.0
WHISPER_WINDOW_FRAMES = 3000
LONG_BATCH = 4


def named_eval_configs(cfg):
    named = split_eval_config_by_name(resolve_eval_data_config(cfg.data))
    if named is None:
        raise SystemExit("validation_ds has no named sets; this script expects dev_clean/dev_other")
    return named


def materialize(sub_cfg, max_samples, drop_filters=False):
    sub_cfg = copy.deepcopy(sub_cfg)
    OmegaConf.set_struct(sub_cfg, False)
    sub_cfg.max_samples = max_samples
    if drop_filters:
        sub_cfg.min_duration = None
        sub_cfg.max_duration = None
        sub_cfg.max_tokens = None
        sub_cfg.max_tps = None
    return materialize_cuts_for_eval(sub_cfg)


def resolve_field(cut, sub_cfg):
    """Per-cut tag override first, then the set-level field -- MELTMapDataset._resolve_text_field."""
    if getattr(cut, "tags", None):
        field = cut.tags.get("text_field")
        if field:
            return str(field)
    return str(_get_config_value(sub_cfg, "text_field", "text"))


def build_items(cuts, sub_cfg):
    """(items sorted longest-first, n_no_reference). Mirrors MELTMapDataset's valid-index scan and order."""
    strict = bool(_get_config_value(sub_cfg, "strict_text_field", False))
    skip = bool(_get_config_value(sub_cfg, "skip_text_field_mismatch", False))
    items, no_ref = [], 0
    for cut in cuts:
        ref = get_text_from_cut(cut, resolve_field(cut, sub_cfg), strict=strict, skip_on_mismatch=skip)
        if not ref or not ref.strip():
            no_ref += 1
            continue
        raw = " ".join(s.text for s in cut.supervisions if s.text)
        items.append({"cut": cut, "id": cut.id, "duration": float(cut.duration),
                      "ref_pnc": ref.strip(), "ref_raw": raw.strip()})
    # MELTMapDataset sorts by duration + custom.num_tokens (absent on validation splits) descending.
    items.sort(key=lambda it: it["duration"], reverse=True)
    return items, no_ref


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        audio = load_audio_from_cut(it["cut"])
        return {"id": it["id"], "audio": audio}


def collate(batch):
    return batch


def _features(processor, audios, device, long_form):
    kw = dict(sampling_rate=SAMPLE_RATE, return_tensors="pt")
    if long_form:
        # Keep everything past 30 s and pad to the longest, with the mask HF's long-form path needs.
        kw.update(truncation=False, padding="longest", return_attention_mask=True)
    fe = processor.feature_extractor(audios, **kw)
    mask = fe.get("attention_mask")
    return fe.input_features.to(device=device, dtype=torch.bfloat16), (mask.to(device) if mask is not None else None)


def decode_set(items, model, processor, device, batch_size, language, num_workers, long_form=False):
    """Greedy-decode `items` (already duration-sorted). Returns ({cut_id: (hypothesis, detected_lang)}, failed_ids).

    `long_form=False` is Whisper's short-form path (input truncated at 30 s, the feature extractor's default);
    `long_form=True` is its sequential long-form algorithm, for cuts that do not fit one window.
    """
    tok = processor.tokenizer
    loader = torch.utils.data.DataLoader(
        AudioDataset(items), batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=collate,
    )
    out, failed = {}, []
    for batch in loader:
        ok = [b for b in batch if b["audio"] is not None]
        failed += [b["id"] for b in batch if b["audio"] is None]
        if not ok:
            continue
        feats, attn = _features(processor, [b["audio"] for b in ok], device, long_form)

        gen_kwargs = dict(task="transcribe", num_beams=1, do_sample=False)
        if long_form:
            gen_kwargs.update(attention_mask=attn, return_timestamps=True, condition_on_prev_tokens=False)
        else:
            gen_kwargs["max_new_tokens"] = 256
        if language:
            gen_kwargs["language"] = language
        with torch.no_grad():
            detected = [None] * len(ok)
            if not language:
                # LID reads the first window only.
                lang_ids = model.detect_language(input_features=feats[..., :WHISPER_WINDOW_FRAMES])
                detected = [t.strip("<|>") for t in tok.convert_ids_to_tokens(lang_ids.reshape(-1).tolist())]
            ids = model.generate(input_features=feats, **gen_kwargs)
        if isinstance(ids, dict) or hasattr(ids, "sequences"):
            ids = ids["sequences"]
        hyps = tok.batch_decode(ids, skip_special_tokens=True)
        for b, h, d in zip(ok, hyps, detected):
            out[b["id"]] = (h.strip(), d)
        logger.info("%d/%d decoded%s", len(out), len(items), " (long-form)" if long_form else "")
    return out, failed


def score(refs, hyps):
    n = sum(len(r.split()) for r in refs)
    if not refs or n == 0:
        return None
    m = jiwer.process_words(refs, hyps)
    return {
        "wer": m.wer,
        "cer": jiwer.cer(refs, hyps),
        "substitution_rate": m.substitutions / n,
        "deletion_rate": m.deletions / n,
        "insertion_rate": m.insertions / n,
        "n_ref_words": n,
    }


def score_view(rows, hyp_of, norm):
    """Score `rows` (items) with hypotheses `hyp_of[id]` against pnc_text (headline) and the raw transcript."""
    hyps = [norm(hyp_of[it["id"]]).strip() for it in rows]
    refs = [norm(it["ref_pnc"]).strip() for it in rows]
    refs_raw = [norm(it["ref_raw"]).strip() for it in rows]

    # jiwer refuses an empty reference; the trainer would fail identically.
    keep = [i for i, r in enumerate(refs) if r]
    res = {"n": len(keep), "n_empty_ref_after_norm": len(refs) - len(keep),
           "pnc_text": score([refs[i] for i in keep], [hyps[i] for i in keep])}
    keep_raw = [i for i in keep if refs_raw[i]]
    res["raw_transcript"] = score([refs_raw[i] for i in keep_raw], [hyps[i] for i in keep_raw])
    # pnc_text against the raw transcript: what the truecase pass changed, after normalisation.
    res["pnc_vs_raw_reference"] = score([refs_raw[i] for i in keep_raw], [refs[i] for i in keep_raw])
    res["n_ref_pnc_differs_from_raw"] = sum(1 for i in keep_raw if refs[i] != refs_raw[i])
    # Numerals are the one normaliser gap that is cheap to count: Whisper writes "1990", the reference spells it out.
    res["n_hyp_with_digit"] = sum(1 for h in hyps if re.search(r"\d", h))
    res["n_ref_with_digit"] = sum(1 for r in refs if re.search(r"\d", r))
    return res, hyps, refs, refs_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="projects/ablation-campaign/ABL-MA-librispeech.yaml")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=500, help="the arm's in-training eval max_samples")
    ap.add_argument("--unforced", action="store_true", help="also decode the full sets with Whisper's own LID")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: keep the first N items per pass")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    configure_logging()
    set_seed(42)
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))
    named = named_eval_configs(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(args.device).eval()
    processor = WhisperProcessor.from_pretrained(args.model)
    logger.info("Loaded %s, attn_implementation=%s, device=%s", args.model, model.config._attn_implementation, args.device)
    norm = BasicTextNormalizer()

    passes = [("forced_en_full", None, "en"), (f"forced_en_first{args.subset}", args.subset, "en")]
    if args.unforced:
        passes.append(("unforced_full", None, None))

    def decode(items, language, long_form, bs):
        return decode_set(items, model, processor, args.device, bs, language, args.num_workers, long_form)

    results, wall0 = {}, time.time()
    for name, sub_cfg in named.items():
        n_unfiltered = len(materialize(sub_cfg, None, drop_filters=True))
        n_arm = len(materialize(sub_cfg, None))
        for pass_name, max_samples, language in passes:
            t0 = time.time()
            cuts = materialize(sub_cfg, max_samples)
            items, no_ref = build_items(cuts, sub_cfg)
            if args.limit:
                items = items[: args.limit]
            short = [it for it in items if it["duration"] <= WHISPER_WINDOW_S]
            long_ = [it for it in items if it["duration"] > WHISPER_WINDOW_S]

            # Short cuts: one Whisper window, decoded once. Cuts past 30 s: the arm's processor windows them and
            # feeds every window to the encoder, so Whisper is given the same thing two ways -- truncated at 30 s
            # (short-form default) and its own long-form algorithm -- and the set is scored under both.
            dec_short, failed = decode(short, language, False, args.batch_size) if short else ({}, [])
            dec_long, failed_l = decode(long_, language, True, LONG_BATCH) if long_ else ({}, [])
            dec_trunc, failed_t = decode(long_, language, False, LONG_BATCH) if long_ else ({}, [])
            failed = failed + failed_l + failed_t
            hyp_long = {k: v[0] for k, v in {**dec_short, **dec_long}.items()}
            hyp_trunc = {k: v[0] for k, v in {**dec_short, **dec_trunc}.items()}

            rows = [it for it in items if it["id"] in hyp_long]
            res, hyps, refs, refs_raw = score_view(rows, hyp_long, norm)
            res["headline_is"] = "every decoded cut; cuts over 30 s use Whisper long-form"
            res["all_cuts_long_over_30s_truncated"] = score_view(rows, hyp_trunc, norm)[0]
            res["within_30s_only"] = score_view([r for r in rows if r["duration"] <= WHISPER_WINDOW_S], hyp_long, norm)[0]
            res["over_30s_only_longform"] = score_view([r for r in rows if r["duration"] > WHISPER_WINDOW_S], hyp_long, norm)[0] if long_ else None
            res["over_30s_only_truncated"] = score_view([r for r in rows if r["duration"] > WHISPER_WINDOW_S], hyp_trunc, norm)[0] if long_ else None
            res.update({
                "cuts_unfiltered_in_shar": n_unfiltered,
                "cuts_after_arm_filters": n_arm,
                "cuts_materialised": len(cuts),
                "cuts_without_reference": no_ref,
                "cuts_audio_load_failed": len(failed),
                "cuts_over_30s": len(long_),
                "cuts_decoded": len(hyp_long),
                "max_duration_s": max((it["duration"] for it in items), default=None),
                "decode_seconds": time.time() - t0,
            })
            if language is None:
                langs = [v[1] for v in dec_short.values()] + [v[1] for v in dec_long.values()]
                res["detected_language_counts"] = {l: langs.count(l) for l in sorted(set(langs))}
            results[f"{name}/{pass_name}"] = res
            with open(out_dir / f"hypotheses_{name}_{pass_name}.jsonl", "w") as f:
                for it, h, r, rr in zip(rows, hyps, refs, refs_raw):
                    f.write(json.dumps({"cut_id": it["id"], "duration": it["duration"], "reference": r,
                                        "reference_raw_transcript": rr, "hypothesis": h,
                                        "hypothesis_unnormalised": hyp_long[it["id"]],
                                        "hypothesis_truncated_unnormalised": hyp_trunc[it["id"]]}) + "\n")
            logger.info("%s/%s: %s", name, pass_name, json.dumps(res))

    # Batch-composition check: the subset scored out of the full-set decode vs. its own decode.
    for name in named:
        full = {}
        for line in open(out_dir / f"hypotheses_{name}_forced_en_full.jsonl"):
            d = json.loads(line)
            full[d["cut_id"]] = d
        sub = [json.loads(line) for line in open(out_dir / f"hypotheses_{name}_forced_en_first{args.subset}.jsonl")]
        ids = [d["cut_id"] for d in sub if d["cut_id"] in full]
        results[f"{name}/subset_scored_from_full_decode"] = (
            (score([full[i]["reference"] for i in ids], [full[i]["hypothesis"] for i in ids]) or {}) | {"n": len(ids)}
        )

    results["wall_seconds_total"] = time.time() - wall0
    results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["config"] = args.config
    results["batch_size"] = args.batch_size
    with open(out_dir / "whisper_reference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
