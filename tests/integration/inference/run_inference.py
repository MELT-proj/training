"""
ASR inference runner for MELT (Multimodal Encoder Language Transformer) models.

Evaluates a MELT checkpoint against a HuggingFace speech dataset or a local
Lhotse SHAR directory.  Results are streamed to a jsonl file so that partial
progress is never lost, and a run can be resumed after interruption.

Usage
-----
    python run_inference.py \\
        --checkpoint ./checkpoints/melt-asr \\
        --dataset openslr/librispeech_asr \\
        --split test.clean \\
        --max-samples 100

    # Local SHAR directory
    python run_inference.py \\
        --checkpoint ./checkpoints/melt-asr \\
        --dataset shar::/data/librispeech/shar/test-clean \\

Output
------
A run-id directory is created under ``--output-dir`` (default: ``./eval_runs``)
containing two files::

    <output-dir>/<run-id>/
    ├── metadata.json    # run parameters, environment, aggregate metrics
    └── results.jsonl    # per-sample predictions (appended incrementally)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from melt.modeling import MELTForCausalLM, MELTProcessor

logger = logging.getLogger(__name__)

# Characters that are safe in directory names (no quoting needed in shells).
_SLUG_SAFE = re.compile(r"[^a-zA-Z0-9._-]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Result for a single audio sample."""

    file_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    input_tokens: list[str] | None = None


@dataclass
class EvalStats:
    """Aggregated evaluation statistics."""

    wer: float
    cer: float
    total_samples: int
    completed_samples: int
    error_samples: int
    skipped_samples: int


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with a plain stream handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Environment diagnostics
# ---------------------------------------------------------------------------


def log_environment_info(device: str) -> None:
    """Log CUDA / GPU details at DEBUG level."""
    logger.debug("PyTorch version: %s", torch.__version__)
    logger.debug("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.debug("Number of GPUs: %d", num_gpus)
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.debug(
                "  GPU %d: %s (%.1f GiB VRAM, compute %d.%d)",
                i,
                props.name,
                props.total_memory / (1024**3),
                props.major,
                props.minor,
            )

    logger.debug("Using device: %s", device)


# ---------------------------------------------------------------------------
# Run identifier
# ---------------------------------------------------------------------------


def _slug(text: str, max_len: int = 60) -> str:
    """Turn *text* into a filesystem-safe token."""
    return _SLUG_SAFE.sub("_", text)[:max_len]


def make_run_id(
    checkpoint: str,
    dataset_spec: str,
    split: str | None,
    max_samples: int | None,
    prompt: str,
) -> str:
    """Build a short, human-readable run identifier from key parameters.

    The prompt is shortened via MD5 to keep the directory name manageable.
    """
    ckpt_slug = _slug(Path(checkpoint).name)

    if dataset_spec.startswith("shar::"):
        ds_slug = _slug(Path(dataset_spec[6:]).name)
    else:
        ds_slug = _slug(dataset_spec.replace("/", "_"))

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]

    parts = [ckpt_slug, ds_slug]
    if split and not dataset_spec.startswith("shar::"):
        parts.append(_slug(split))
    if max_samples is not None:
        parts.append(f"n{max_samples}")
    parts.append(prompt_hash)

    return "_".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inference script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a MELT checkpoint on a speech dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # LibriSpeech test-clean
            python run_inference.py \\
                --checkpoint ./ckpts/melt-asr \\
                --dataset openslr/librispeech_asr \\
                --split test.clean \\
                --max-samples 200

            # Local SHAR directory
            python run_inference.py \\
                --checkpoint ./ckpts/melt-asr \\
                --dataset shar::/data/librispeech/shar/test-clean

            # Resume an interrupted run
            python run_inference.py \\
                --checkpoint ./ckpts/melt-asr \\
                --dataset openslr/librispeech_asr \\
                --split test.clean \\
                --max-samples 200

            # Force re-run from scratch
            python run_inference.py \\
                --checkpoint ./ckpts/melt-asr \\
                --dataset openslr/librispeech_asr \\
                --overwrite
        """),
    )

    # --- Required ---
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the MELT checkpoint directory.",
    )
    parser.add_argument(
        "--processor", type=str, default=None,
        help="Path to the MELT processor directory (default: same as --checkpoint).",
    )

    # --- Dataset ---
    parser.add_argument(
        "--dataset", type=str, default="openslr/librispeech_asr",
        help="HuggingFace dataset name (e.g. 'openslr/librispeech_asr') or "
             "local SHAR directory prefixed with 'shar::' "
             "(e.g. 'shar::/data/librispeech/shar/test-clean').",
    )
    parser.add_argument(
        "--split", type=str, default="test.clean",
        help="Dataset split to evaluate on (ignored for SHAR data; default: test.clean).",
    )
    parser.add_argument(
        "--text-column", type=str, default="text",
        help="Dataset column with reference transcriptions (default: text).",
    )
    parser.add_argument(
        "--file-id-column", type=str, default="id",
        help="Dataset column for per-sample identifiers (default: id).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to the first N samples (default: all).",
    )

    # --- Generation ---
    parser.add_argument(
        "--prompt", type=str,
        default="<|audio|>Transcribe this audio.",
        help="Prompt template (default: '<|audio|>Transcribe this audio.').",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Omit for greedy decoding.",
    )

    # --- Output ---
    parser.add_argument(
        "--apply-chat-template", action="store_true",
        help="Wrap the prompt with the tokenizer's chat template before generation. "
             "When set, '{audio_token}' in --prompt is replaced by the processor's "
             "audio token and the result is formatted via apply_chat_template().",
    )
    parser.add_argument(
        "--verbose-out", action="store_true",
        help="Include tokenized input prompt ('input_tokens' field) in results.jsonl.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_runs",
        help="Parent directory for run-specific output folders (default: ./eval_runs).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run identifier (auto-generated from parameters if not set).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete any existing run directory and start from scratch.",
    )

    # --- Misc ---
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging (logs GPU info, batch sizes, etc.).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser


# ---------------------------------------------------------------------------
# Model & processor I/O
# ---------------------------------------------------------------------------


def load_model(checkpoint: str, device: str) -> MELTForCausalLM:
    """Load a MELTForCausalLM from *checkpoint* and move it to *device*."""
    logger.info("Loading model from %s ...", checkpoint)
    model = MELTForCausalLM.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def load_processor(processor_path: str) -> MELTProcessor:
    """Load the MELTProcessor from *processor_path*."""
    logger.info("Loading processor from %s ...", processor_path)
    return MELTProcessor.from_pretrained(processor_path)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _resolve_audio(sample: dict[str, Any]) -> tuple[Any, int]:
    """Extract (audio_array, sample_rate) from a dataset row."""
    audio_col = sample.get("audio")
    if isinstance(audio_col, dict):
        return audio_col["array"], audio_col["sampling_rate"]
    return audio_col, 16000


def _resolve_dataset(
    dataset_spec: str,
    split: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Resolve ``--dataset`` into a list of sample dicts.

    Supports two formats:

    * **HuggingFace dataset** — standard HF dataset name.
    * **Local SHAR directory** — ``"shar::/path/to/shar/dir"``.
    """
    if dataset_spec.startswith("shar::"):
        return _load_shar_dataset(dataset_spec[len("shar::"):], max_samples)
    else:
        return _load_hf_dataset(dataset_spec, split, max_samples)


def _load_hf_dataset(
    dataset_name: str,
    split: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset split into a list of sample dicts."""
    from datasets import load_dataset as hf_load_dataset

    logger.info("Loading HF dataset %s [%s] ...", dataset_name, split)
    dataset = hf_load_dataset(dataset_name, split=split, trust_remote_code=True)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info("HF dataset ready: %d samples.", len(dataset))
    return list(dataset)


def _load_shar_dataset(
    shar_path: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Load a Lhotse CutSet from a local SHAR directory.

    Each cut is normalised to a dict with keys ``id``, ``audio``, and
    ``text`` so the inference loop works identically to the HF path.
    """
    from lhotse import CutSet

    shar_path_expanded = os.path.expanduser(os.path.expandvars(shar_path))
    if not Path(shar_path_expanded).exists():
        raise FileNotFoundError(
            f"SHAR directory not found: {shar_path_expanded}"
        )

    logger.info("Loading CutSet from SHAR: %s", shar_path_expanded)
    cuts = CutSet.from_shar(in_dir=shar_path_expanded)
    logger.info("CutSet ready: %d cuts.", len(cuts))

    samples: list[dict[str, Any]] = []
    for i, cut in enumerate(tqdm(cuts, desc="Loading SHAR audio", unit="cut")):
        if max_samples is not None and i >= max_samples:
            break

        audio_array = cut.load_audio()
        sr: int = getattr(cut, "sampling_rate", 16000)
        text: str = cut.supervisions[0].text if cut.supervisions else ""

        samples.append({
            "id": cut.id,
            "audio": {"array": audio_array, "sampling_rate": sr},
            "text": text,
        })

    logger.info("Loaded %d samples from SHAR directory.", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    """Compute corpus-level WER and CER via ``jiwer``.

    Returns (0.0, 0.0) when jiwer is not installed.
    """
    try:
        from jiwer import cer, wer  # type: ignore[import-untyped]
        return wer(references, hypotheses), cer(references, hypotheses)
    except ImportError:
        logger.warning("jiwer is not installed — returning 0.0 for WER/CER.")
        return 0.0, 0.0


def _single_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Compute WER / CER for a single utterance pair."""
    try:
        from jiwer import cer, wer  # type: ignore[import-untyped]
        return wer([reference], [hypothesis]), cer([reference], [hypothesis])
    except ImportError:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------


def load_completed_ids(results_path: Path) -> set[str]:
    """Read the set of already-processed sample IDs from a jsonl file."""
    if not results_path.exists():
        return set()

    ids: set[str] = set()
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                file_id = obj.get("id") or obj.get("file_id")
                if file_id:
                    ids.add(file_id)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", results_path)
    return ids


def _append_result(results_path: Path, result: SampleResult) -> None:
    """Append a single result line to *results_path* and flush immediately."""
    obj: dict[str, Any] = {
        "id": result.file_id,
        "reference": result.reference,
        "hypothesis": result.hypothesis,
        "wer": round(result.wer, 6),
        "cer": round(result.cer, 6),
    }
    if result.input_tokens is not None:
        obj["input_tokens"] = result.input_tokens
    line = json.dumps(obj)
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _save_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write the metadata dictionary to *metadata_path* as JSON."""
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
        fh.write("\n")


def _load_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """Read existing metadata.json, or return None."""
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse %s — ignoring.", metadata_path)
        return None


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def run_inference(
    model: MELTForCausalLM,
    processor: MELTProcessor,
    dataset: list[dict[str, Any]],
    *,
    prompt_template: str,
    text_column: str,
    file_id_column: str,
    device: str,
    max_new_tokens: int,
    temperature: float | None,
    results_path: Path,
    completed_ids: set[str],
    verbose_out: bool = False,
    apply_chat_template: bool = False,
) -> EvalStats:
    """Run inference over *dataset*, streaming results to *results_path*.

    Samples whose ``file_id_column`` value appears in *completed_ids* are
    skipped so that an interrupted run can be resumed safely.

    Returns aggregate :class:`EvalStats`.
    """

    references: list[str] = []
    hypotheses: list[str] = []
    completed = 0
    errors = 0
    skipped = 0

    # --- Prompt formatting ------------------------------------------------
    # Replace {audio_token} placeholder with the processor's audio token.
    text_prompt = prompt_template.replace("{audio_token}", processor.audio_token)

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
        logger.debug("Chat-template formatted prompt: %s", text_prompt)

    # ------------------------------------------------------------------
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature is not None,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if temperature is not None:
        generate_kwargs["temperature"] = temperature

    for instance in tqdm(dataset, desc="Running ASR inference", unit="sample"):
        file_id = str(instance[file_id_column])

        if file_id in completed_ids:
            skipped += 1
            continue

        try:
            reference_text = str(instance[text_column]).lower().strip()
            audio_array, sample_rate = _resolve_audio(instance)

            # --- Preprocessing ---------------------------------------------
            processor_outputs = processor(
                text=text_prompt,
                audio=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                k: v.to(device)
                for k, v in processor_outputs.items()
                if v is not None
            }

            logger.debug(
                "Sample '%s': input_ids shape=%s, input_features shape=%s",
                file_id,
                tuple(inputs["input_ids"].shape),
                tuple(inputs.get("input_features", torch.tensor(0.0)).shape),
            )

            # --- Generation ------------------------------------------------
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    input_features=inputs.get("input_features"),
                    features_attention_mask=inputs.get("features_attention_mask"),
                    attention_mask=inputs.get("attention_mask"),
                    **generate_kwargs,
                )

            prompt_len = inputs["input_ids"].shape[1]
            hypothesis = processor.batch_decode(
                generated_ids[:, prompt_len:],
                skip_special_tokens=True,
            )[0].strip().lower()

            row_wer, row_cer = _single_wer_cer(reference_text, hypothesis)

            input_tokens = None
            if verbose_out:
                input_tokens = processor.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0]
                )

            result = SampleResult(
                file_id=file_id,
                reference=reference_text,
                hypothesis=hypothesis,
                wer=row_wer,
                cer=row_cer,
                input_tokens=input_tokens,
            )

            _append_result(results_path, result)

            references.append(reference_text)
            hypotheses.append(hypothesis)
            completed += 1

        except Exception:
            logger.exception(
                "Error processing sample '%s' — skipping.", file_id
            )
            errors += 1
            continue

    corpus_wer, corpus_cer = compute_wer_cer(references, hypotheses)
    return EvalStats(
        wer=corpus_wer,
        cer=corpus_cer,
        total_samples=len(dataset),
        completed_samples=completed,
        error_samples=errors,
        skipped_samples=skipped,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be a positive integer.")

    # ------------------------------------------------------------------
    # Device & environment
    # ------------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_environment_info(device)

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Model & processor
    # ------------------------------------------------------------------
    model = load_model(args.checkpoint, device)
    processor_path = args.processor or args.checkpoint
    processor = load_processor(processor_path)

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    run_id = args.run_id or make_run_id(
        checkpoint=args.checkpoint,
        dataset_spec=args.dataset,
        split=args.split if not args.dataset.startswith("shar::") else None,
        max_samples=args.max_samples,
        prompt=args.prompt,
    )
    run_dir = Path(args.output_dir) / run_id
    results_path = run_dir / "results.jsonl"
    metadata_path = run_dir / "metadata.json"

    if args.overwrite and run_dir.exists():
        import shutil
        logger.info("--overwrite: removing existing run directory %s", run_dir)
        shutil.rmtree(str(run_dir))

    is_resume = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)

    now_ts = datetime.now(timezone.utc)

    # --- Metadata baseline ------------------------------------------------
    gpu_names: list[str] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_properties(i).name)

    existing_meta = _load_metadata(metadata_path)
    if existing_meta is not None and is_resume:
        created_at = existing_meta.get("created_at", now_ts.isoformat())
        restarts: list[str] = existing_meta.get("restarts", [])
        restarts.append(now_ts.isoformat())
    else:
        created_at = now_ts.isoformat()
        restarts = []

    base_metadata: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": os.path.abspath(args.checkpoint),
        "processor": os.path.abspath(processor_path),
        "dataset": args.dataset,
        "split": args.split if not args.dataset.startswith("shar::") else None,
        "max_samples": args.max_samples,
        "prompt": args.prompt,
        "prompt_hash": hashlib.md5(args.prompt.encode()).hexdigest()[:8],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": device,
        "num_gpus": len(gpu_names),
        "gpu_names": gpu_names,
        "created_at": created_at,
        "restarts": restarts,
        "status": "RUNNING",
    }
    _save_metadata(metadata_path, base_metadata)

    # ------------------------------------------------------------------
    # Resume or start fresh
    # ------------------------------------------------------------------
    completed_ids: set[str] = set()
    if is_resume and results_path.exists():
        completed_ids = load_completed_ids(results_path)
        if completed_ids:
            logger.info(
                "Resuming: %d samples already completed, will skip them.",
                len(completed_ids),
            )
        else:
            logger.info("Existing results file is empty — starting fresh.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = _resolve_dataset(args.dataset, args.split, args.max_samples)
    logger.info("Dataset ready: %d samples.", len(dataset))

    if completed_ids:
        pending = len(dataset) - len(completed_ids)
        logger.info(
            "After skipping completed: %d samples remaining.", pending
        )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    start_time = time.time()
    start_dt = datetime.now(timezone.utc)

    try:
        stats = run_inference(
            model=model,
            processor=processor,
            dataset=dataset,
            prompt_template=args.prompt,
            text_column=args.text_column,
            file_id_column=args.file_id_column,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            results_path=results_path,
            completed_ids=completed_ids,
            verbose_out=args.verbose_out,
            apply_chat_template=args.apply_chat_template,
        )
    except Exception:
        # Update metadata to CRASHED before exiting
        crash_meta = _load_metadata(metadata_path) or base_metadata
        crash_meta["status"] = "CRASHED"
        crash_meta["end_time"] = datetime.now(timezone.utc).isoformat()
        _save_metadata(metadata_path, crash_meta)
        logger.exception("Run crashed — metadata updated.")
        raise

    end_time = time.time()
    end_dt = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Final metadata
    # ------------------------------------------------------------------
    final_metadata = _load_metadata(metadata_path) or base_metadata
    final_metadata.update({
        "status": "COMPLETED",
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_seconds": round(end_time - start_time, 2),
        "total_samples": stats.total_samples,
        "completed_samples": stats.completed_samples,
        "error_samples": stats.error_samples,
        "skipped_samples": stats.skipped_samples,
        "wer": round(stats.wer, 6),
        "cer": round(stats.cer, 6),
    })
    _save_metadata(metadata_path, final_metadata)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    logger.info(
        "WER: %.4f  |  CER: %.4f  |  completed: %d  |  errors: %d  |  skipped: %d",
        stats.wer,
        stats.cer,
        stats.completed_samples,
        stats.error_samples,
        stats.skipped_samples,
    )
    logger.info("Results written to %s", run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
