"""
Diagnostic script: compare logits from forward() vs the prefill step of generate().

This tool helps determine whether MELTForCausalLM.generate() constructs the
same input tensors as MELTForCausalLM.forward().  If the prefill logits
diverge, the problem lies in how generate() prepares inputs for the
underlying text decoder.  If they agree, the issue is downstream (e.g.
attention-mask extension during autoregressive decoding, or sampling).

Usage
-----
    # Compare using synthetic (sine-wave) audio — no external file needed
    python diagnose_logits.py --checkpoint ./checkpoints/melt-asr --synthetic

    # Compare using a real audio file
    python diagnose_logits.py --checkpoint ./checkpoints/melt-asr          \\
                             --audio-file /path/to/audio.wav               \\
                             --prompt "<|audio|> Transcribe this audio."

    # Compare using a sample from a Lhotse SHAR directory (e.g. voxpopuli)
    python diagnose_logits.py                                              \\
        --checkpoint $SCRATCH/melt-data/outputs/SFT-v1.2.7.2-64nodes       \\
        --shar-dir /mnt/scratch-nyx/giuseppe/melt/melt-data/shar/voxpopuli/en/test \\
        --apply-chat-template                                              \\
        --prompt "<|audio|> Transcribe this audio."

    # Compare using a checkpoint that needs a processor from a different path
    python diagnose_logits.py --checkpoint ./checkpoints/melt-asr          \\
                             --processor ./checkpoints/melt-processor       \\
                             --synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

from melt.modeling import MELTForCausalLM, MELTProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare forward() logits against generate() prefill logits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )

    # --- Model ---
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the MELT checkpoint directory.",
    )
    parser.add_argument(
        "--processor", type=str, default=None,
        help="Path to the MELT processor directory (default: same as --checkpoint).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: cuda if available, else cpu).",
    )

    # --- Input ---
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic sine-wave audio tone instead of a real file.",
    )
    parser.add_argument(
        "--audio-file", type=str, default=None,
        help="Path to an audio file to use as input.",
    )
    parser.add_argument(
        "--shar-dir", type=str, default=None,
        help="Path to a Lhotse SHAR directory.  The first cut is used as input.",
    )
    parser.add_argument(
        "--prompt", type=str, default="<|audio|> Transcribe this audio.",
        help="Text prompt containing the {audio_token} placeholder.",
    )
    parser.add_argument(
        "--apply-chat-template", action="store_true",
        help="Wrap the prompt with the tokenizer's chat template before processing.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Sampling rate for synthetic audio (default: 16000).",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="Duration in seconds for synthetic audio (default: 2.0).",
    )
    parser.add_argument(
        "--freq", type=float, default=440.0,
        help="Frequency in Hz for synthetic sine wave (default: 440.0).",
    )

    # --- Comparison ---
    parser.add_argument(
        "--num-tokens", type=int, default=5,
        help="Number of tokens to generate and compare (default: 5).",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="Absolute tolerance for allclose comparison (default: 1e-4).",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3,
        help="Relative tolerance for allclose comparison (default: 1e-3).",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Write detailed per-metric results to a JSON file.",
    )

    return parser


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def make_synthetic_audio(
    duration: float = 2.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
) -> np.ndarray:
    """Generate a sine-wave audio array (mono, float32)."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    logger.info(
        "Synthetic audio: %.1f s @ %d Hz, tone=%.0f Hz → %d samples",
        duration, sample_rate, freq, num_samples,
    )
    return audio


def load_audio_file(path: str) -> np.ndarray:
    """Load a mono float32 audio array from *path* via soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit(
            "soundfile is required to load audio files. "
            "Install it with `pip install soundfile` or use --synthetic."
        )
    audio, _sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    logger.info("Loaded audio from %s: %d samples", path, len(audio))
    return audio


def load_shar_sample(shar_dir: str) -> tuple[np.ndarray, int, str]:
    """Load the first cut from a Lhotse SHAR directory.

    Returns
    -------
    (audio_array, sample_rate, reference_text)
    """
    try:
        from lhotse import CutSet
    except ImportError:
        raise SystemExit(
            "lhotse is required to load SHAR data. "
            "Install it with `pip install lhotse` or use --synthetic / --audio-file."
        )

    shar_path = Path(shar_dir).expanduser().resolve()
    if not shar_path.exists():
        raise FileNotFoundError(f"SHAR directory not found: {shar_path}")

    logger.info("Loading CutSet from SHAR: %s", shar_path)
    cuts = CutSet.from_shar(in_dir=str(shar_path))
    cut = cuts[0]
    logger.info("Using cut: id=%s  duration=%.1fs", cut.id, cut.duration)

    audio_array = cut.load_audio()
    sr: int = getattr(cut, "sampling_rate", 16000)
    ref_text: str = cut.supervisions[0].text if cut.supervisions else ""

    logger.info(
        "  sample_rate=%d  shape=%s  dtype=%s  ref_text=%s",
        sr,
        audio_array.shape,
        getattr(audio_array, "dtype", "?"),
        ref_text[:80],
    )
    return audio_array, sr, ref_text


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

_TOP_K = 10


def _step_metrics(
    fwd_logit: torch.Tensor,   # (vocab_size,)
    gen_logit: torch.Tensor,   # (vocab_size,)
    atol: float,
    rtol: float,
) -> dict:
    """Compare two logit vectors and return a dict of metrics."""
    fwd = fwd_logit.float()
    gen = gen_logit.float()

    abs_diff = (fwd - gen).abs()
    allclose = bool(torch.allclose(fwd, gen, atol=atol, rtol=rtol))

    fwd_pred = fwd.argmax().item()
    gen_pred = gen.argmax().item()

    fwd_topk = fwd.topk(_TOP_K).indices.tolist()
    gen_topk = gen.topk(_TOP_K).indices.tolist()

    return {
        "allclose": allclose,
        "token_match": fwd_pred == gen_pred,
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                fwd.unsqueeze(0), gen.unsqueeze(0)
            ).item()
        ),
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "topk_overlap": len(set(fwd_topk) & set(gen_topk)),
        "forward_pred_token": fwd_pred,
        "generate_pred_token": gen_pred,
        "forward_topk_tokens": fwd_topk,
        "generate_topk_tokens": gen_topk,
    }


@torch.no_grad()
def compare(
    model: MELTForCausalLM,
    processor: MELTProcessor,
    text_prompt: str,
    audio_array: np.ndarray,
    sample_rate: int,
    device: str,
    atol: float,
    rtol: float,
    num_tokens: int = 5,
    apply_chat_template: bool = False,
) -> dict:
    """Run forward() and generate() and compare logits across N steps.

    For each generation step *k* (0-indexed):
      1. ``generate()`` produces logit *k* used to sample token *k*.
      2. ``forward()`` is called with the original prompt plus the first *k*
         generated tokens appended, and ``logits[-1]`` is extracted.
      3. The two logit vectors are compared.

    If the logits match at step 0 but diverge later, the problem is in the
    autoregressive loop (attention-mask extension or KV-cache handling).
    """
    # --- Pre-process ----------------------------------------------------------
    text_prompt = text_prompt.replace("{audio_token}", processor.audio_token)

    if apply_chat_template:
        tokenizer = processor.tokenizer
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "--apply-chat-template is set but the tokenizer does not "
                "support apply_chat_template()."
            )
        text_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        logger.info("Chat-template formatted prompt: %s", text_prompt[:120])

    processor_outputs = processor(
        text=text_prompt,
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    prompt_inputs = {
        k: v.to(device) for k, v in processor_outputs.items() if v is not None
    }

    logger.info("Prompt input shapes:")
    for k, v in prompt_inputs.items():
        logger.info("  %s: %s", k, tuple(v.shape))

    # --- generate() with max_new_tokens = num_tokens --------------------------
    generate_out = model.generate(
        input_ids=prompt_inputs["input_ids"],
        input_features=prompt_inputs.get("input_features"),
        features_attention_mask=prompt_inputs.get("features_attention_mask"),
        attention_mask=prompt_inputs.get("attention_mask"),
        max_new_tokens=num_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    # gen_logits: tuple of num_tokens tensors, each (1, vocab_size)
    gen_logits = generate_out.logits
    gen_ids = generate_out.sequences[0]  # (prompt_len + num_tokens,)
    generated_ids = gen_ids[-num_tokens:]  # (num_tokens,)

    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    logger.info(
        "Generated (%d tokens): %s",
        num_tokens,
        decoded[:120],
    )

    # --- Per-step comparison --------------------------------------------------
    steps = []
    for step_idx in range(num_tokens):
        gen_logit = gen_logits[step_idx][0, :]  # (vocab_size,)

        # Build forward() inputs: prompt + first *step_idx* generated tokens
        if step_idx == 0:
            extended_ids = prompt_inputs["input_ids"]
            extended_mask = prompt_inputs.get("attention_mask")
        else:
            prefix = generated_ids[:step_idx].unsqueeze(0)  # (1, step_idx)
            extended_ids = torch.cat([prompt_inputs["input_ids"], prefix], dim=1)
            if prompt_inputs.get("attention_mask") is not None:
                ones = torch.ones(
                    (1, step_idx),
                    dtype=prompt_inputs["attention_mask"].dtype,
                    device=device,
                )
                extended_mask = torch.cat(
                    [prompt_inputs["attention_mask"], ones], dim=1
                )
            else:
                extended_mask = None

        fwd_out = model(
            input_ids=extended_ids,
            input_features=prompt_inputs.get("input_features"),
            features_attention_mask=prompt_inputs.get("features_attention_mask"),
            attention_mask=extended_mask,
            return_dict=True,
            use_cache=False,  # fresh forward, no cache
        )
        fwd_logit = fwd_out.logits[0, -1, :]  # (vocab_size,)

        step_metrics = _step_metrics(fwd_logit, gen_logit, atol=atol, rtol=rtol)
        step_metrics["step"] = step_idx
        steps.append(step_metrics)

        logger.debug(
            "Step %d: allclose=%s  cos=%.8f  max_diff=%.2e  tokens: fwd=%s gen=%s",
            step_idx,
            step_metrics["allclose"],
            step_metrics["cosine_similarity"],
            step_metrics["max_abs_diff"],
            step_metrics["forward_pred_token"],
            step_metrics["generate_pred_token"],
        )

    # --- Aggregate summary ----------------------------------------------------
    allclose_all = all(s["allclose"] for s in steps)
    token_match_all = all(s["token_match"] for s in steps)
    max_abs = max(s["max_abs_diff"] for s in steps)
    mean_abs = sum(s["mean_abs_diff"] for s in steps) / len(steps)
    min_cos = min(s["cosine_similarity"] for s in steps)

    first_divergent = None
    for s in steps:
        if not s["allclose"] or not s["token_match"]:
            first_divergent = s["step"]
            break

    return {
        "num_tokens": num_tokens,
        "atol": atol,
        "rtol": rtol,
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": decoded,
        "steps": steps,
        "summary": {
            "allclose_all_steps": allclose_all,
            "token_match_all_steps": token_match_all,
            "max_abs_diff_across_steps": max_abs,
            "mean_abs_diff_across_steps": mean_abs,
            "min_cosine_similarity": min_cos,
            "first_divergent_step": first_divergent,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PASS_FAIL = {True: "PASS", False: "FAIL"}


def report(metrics: dict) -> None:
    """Pretty-print per-step and summary comparison results."""
    steps = metrics["steps"]
    summary = metrics["summary"]

    # --- Per-step table -------------------------------------------------------
    header = f"  Per-step logit comparison  ({len(steps)} tokens)  "
    rule = "=" * len(header)
    logger.info(rule)
    logger.info(header)
    logger.info(rule)

    # Column headers
    logger.info(
        "  %4s  %6s  %6s  %8s  %9s  %9s  %6s",
        "Step", "Allcl", "Tok=?", "CosSim", "Max|Δ|", "Mean|Δ|", "Top10",
    )
    logger.info("  " + "-" * (len(header) - 2))

    for s in steps:
        logger.info(
            "  %4d  %6s  %6s  %8.6f  %9.2e  %9.2e  %4d/10",
            s["step"],
            PASS_FAIL[s["allclose"]],
            PASS_FAIL[s["token_match"]],
            s["cosine_similarity"],
            s["max_abs_diff"],
            s["mean_abs_diff"],
            s["topk_overlap"],
        )
        # On first failure, print top-k details for diagnosis
        if not s["allclose"] or not s["token_match"]:
            logger.info(
                "       fwd_tok=%s  gen_tok=%s",
                s["forward_pred_token"],
                s["generate_pred_token"],
            )
            logger.info("       fwd top-10: %s", s["forward_topk_tokens"])
            logger.info("       gen top-10: %s", s["generate_topk_tokens"])

    logger.info(rule)

    # --- Summary --------------------------------------------------------------
    logger.info("  Summary:")
    logger.info(
        "    Allclose all steps:          %s",
        PASS_FAIL[summary["allclose_all_steps"]],
    )
    logger.info(
        "    Greedy token match all steps:%s",
        PASS_FAIL[summary["token_match_all_steps"]],
    )
    logger.info(
        "    Min cosine similarity:       %.8f",
        summary["min_cosine_similarity"],
    )
    logger.info(
        "    Max |Δ| across steps:        %.2e",
        summary["max_abs_diff_across_steps"],
    )
    logger.info(
        "    Mean |Δ| across steps:       %.2e",
        summary["mean_abs_diff_across_steps"],
    )
    if summary["first_divergent_step"] is not None:
        logger.info(
            "    First divergent step:       %d",
            summary["first_divergent_step"],
        )

    # --- Generated text for context -------------------------------------------
    logger.info("  Generated: %s", metrics["generated_text"][:100])

    # --- Diagnosis ------------------------------------------------------------
    summary = metrics["summary"]
    steps = metrics["steps"]
    allclose_all = summary["allclose_all_steps"]
    token_match_all = summary["token_match_all_steps"]
    all_top10 = all(s["topk_overlap"] == _TOP_K for s in steps)
    first = summary["first_divergent_step"]

    # The meaningful signal: token predictions and cosine similarity.
    # raw allclose can fail due to float32 noise over large vocabularies.
    if token_match_all and all_top10:
        logger.info(
            "\n✓  Logits match across all %d steps.\n"
            "   Greedy tokens and top-%d rankings are identical.%s\n"
            "   → The generate() path is consistent with forward().\n"
            "   → If generation output is still wrong, the issue is NOT in\n"
            "     the generate/forward wiring — look at the model itself,\n"
            "     the prompt format, or decoding parameters.",
            len(steps),
            _TOP_K,
            "" if allclose_all else (
                " Tiny float32 differences exist (max %.2e) but are\n"
                "   harmless noise." % summary["max_abs_diff_across_steps"]
            ),
        )
    elif first == 0:
        logger.warning(
            "\n⚠  Logits diverge at STEP 0 (prefill).\n"
            "   → generate() constructs different inputs than forward().\n"
            "   → Check: attention_mask shape/dtype, position_ids,\n"
            "     or leftover kwargs passed to text_decoder in forward()."
        )
    else:
        logger.warning(
            "\n⚠  Logits match at step 0 but diverge at step %d.\n"
            "   → The prefill is correct; the problem is in the\n"
            "     autoregressive loop.\n"
            "   → Likely cause: attention-mask extension or\n"
            "     KV-cache position tracking with left-padded inputs.",
            first,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Device ---------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Audio ----------------------------------------------------------------
    if args.synthetic:
        audio_array = make_synthetic_audio(
            duration=args.duration,
            sample_rate=args.sample_rate,
            freq=args.freq,
        )
        sample_rate = args.sample_rate
    elif args.shar_dir:
        audio_array, sample_rate, ref_text = load_shar_sample(args.shar_dir)
        logger.info("Reference text from SHAR: %s", ref_text[:120])
    elif args.audio_file:
        audio_array = load_audio_file(args.audio_file)
        sample_rate = args.sample_rate  # assume this matches
    else:
        raise SystemExit(
            "One of --synthetic, --shar-dir, or --audio-file must be specified."
        )

    # --- Model & processor ----------------------------------------------------
    logger.info("Loading model from %s ...", args.checkpoint)
    model = MELTForCausalLM.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()

    processor_path = args.processor or args.checkpoint
    logger.info("Loading processor from %s ...", processor_path)
    processor = MELTProcessor.from_pretrained(processor_path)

    # --- Run comparison -------------------------------------------------------
    logger.info("Running forward() vs generate() comparison ...")
    metrics = compare(
        model=model,
        processor=processor,
        text_prompt=args.prompt,
        audio_array=audio_array,
        sample_rate=sample_rate,
        device=device,
        atol=args.atol,
        rtol=args.rtol,
        num_tokens=args.num_tokens,
        apply_chat_template=args.apply_chat_template,
    )

    # --- Report ---------------------------------------------------------------
    report(metrics)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics written to %s", json_path)

    # --- Exit code ------------------------------------------------------------
    if not metrics["summary"]["allclose_all_steps"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
