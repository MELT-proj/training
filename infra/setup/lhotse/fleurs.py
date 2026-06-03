"""Convert FLEURS dataset from HuggingFace to Lhotse Shar format.

This script downloads the FLEURS dataset from HuggingFace and converts
it to Lhotse Shar archives for efficient training using multiprocessing.

Dataset Structure:
------------------
FLEURS is a multilingual dataset where:
- config: language code (e.g., "en_us", "de_de", "fr_fr")
- splits: "train", "validation", "test"

Output Directory Structure:
--------------------------
    {BASE_OUTPUT_DIR}/fleurs/{lang}/{split}/
    Example: shar/fleurs/en_us/train/

Reference: https://huggingface.co/datasets/google/fleurs
"""

import argparse
import logging
import os
from pathlib import Path

from batch_utils import convert_subset_to_shar_batched


# --- Configuration ---
DATASET_NAME = "google/fleurs"
DATASET_NICKNAME = "fleurs"
BASE_OUTPUT_DIR = Path(os.environ.get("LHOTSE_DATA_SHAR_ROOT", "/mnt/home/giuseppe/myscratch/melt-data/shar"))
SHARD_SIZE = 4000
AUDIO_FORMAT = "flac"
MARKER_ROOT = BASE_OUTPUT_DIR / ".conversion_markers"

# Default languages to convert (can be overridden via CLI)
DEFAULT_LANGUAGES = [
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
]

# Default splits
DEFAULT_SPLITS = ["train", "validation", "test"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_language_code(lang: str) -> str:
    """Extract base language code from locale (e.g., 'en_us' -> 'en')."""
    return lang.split("_")[0]


def get_output_dir(lang: str, split: str) -> Path:
    """Construct output directory path."""
    return BASE_OUTPUT_DIR / DATASET_NICKNAME / lang / split


def _marker_path_for_output(output_dir: Path) -> Path:
    try:
        rel = output_dir.relative_to(BASE_OUTPUT_DIR)
    except Exception:
        rel = Path(output_dir.name)
    marker = MARKER_ROOT / rel
    marker.parent.mkdir(parents=True, exist_ok=True)
    return marker.with_suffix(".done")


def is_conversion_complete(output_dir: Path) -> bool:
    """Check if conversion is already complete."""
    marker = _marker_path_for_output(output_dir)
    if marker.exists():
        return True
    return output_dir.exists() and any(output_dir.iterdir())


def mark_conversion_complete(output_dir: Path, count: int, errors: int) -> None:
    marker_path = _marker_path_for_output(output_dir)
    marker_path.write_text(f"Conversion completed successfully.\nCuts processed: {count}\nErrors: {errors}\n")
    logger.info(f"Created completion marker: {marker_path}")


def convert_one(
    lang: str,
    split: str,
    force: bool = False,
    num_workers: int | None = None,
    hf_num_proc: int = 4,
) -> tuple[int, int] | tuple[None, None]:
    """Convert a single language/split combination to Shar format.

    Args:
        lang: Language code (e.g., "en_us", "de_de").
        split: Data split (e.g., "train", "validation", "test").
        force: If True, re-run conversion even if already complete.
        num_workers: Number of parallel workers (default: cpu_count).
        hf_num_proc: Number of HuggingFace loading processes.

    Returns:
        Tuple of (count, errors) if processed, or (None, None) if skipped.
    """
    output_dir = get_output_dir(lang, split)

    # Check if conversion is already complete
    if not force and is_conversion_complete(output_dir):
        marker = _marker_path_for_output(output_dir)
        logger.info(f"SKIPPING {lang}/{split} - already complete (marker: {marker})")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {lang}/{split} to: {output_dir}")

    # Use the batched converter with multiprocessing
    count, errors = convert_subset_to_shar_batched(
        dataset_name=DATASET_NAME,
        hf_config=lang,
        hf_split=split,
        output_dir=output_dir,
        audio_format=AUDIO_FORMAT,
        shard_size=SHARD_SIZE,
        language=_get_language_code(lang),
        num_workers=num_workers,
        hf_num_proc=hf_num_proc,
        text_field="transcription",  # FLEURS uses "transcription" field
    )

    logger.info(f"Finished {lang}/{split}! Processed {count} cuts with {errors} errors.")

    # Mark conversion as complete only if no errors
    if errors == 0:
        mark_conversion_complete(output_dir, count, errors)
    else:
        logger.warning(f"Not marking {lang}/{split} as complete due to {errors} errors")

    return count, errors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert FLEURS dataset from HuggingFace to Lhotse Shar format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert default languages:
  python fleurs.py

  # Convert specific languages:
  python fleurs.py --configs en_us de_de fr_fr

  # Custom number of workers:
  python fleurs.py --num-workers 8

  # Force re-conversion:
  python fleurs.py --force
        """,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=f"Language codes to convert (default: {DEFAULT_LANGUAGES}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to convert (default: {DEFAULT_SPLITS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even for already completed subsets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for conversion (default: cpu_count).",
    )
    parser.add_argument(
        "--hf-num-proc",
        type=int,
        default=4,
        help="Number of HuggingFace data loading processes (default: 4).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, ...).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    configs = args.configs if args.configs else DEFAULT_LANGUAGES

    logger.info(f"Base output directory: {BASE_OUTPUT_DIR / DATASET_NICKNAME}")
    logger.info(f"Languages: {configs}")
    logger.info(f"Splits: {args.splits}")

    total_count = 0
    total_errors = 0
    skipped = 0

    for lang in configs:
        for split in args.splits:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {lang}/{split}")
            logger.info(f"{'=' * 60}")

            count, errors = convert_one(
                lang,
                split,
                force=args.force,
                num_workers=args.num_workers,
                hf_num_proc=args.hf_num_proc,
            )

            if count is None:
                skipped += 1
            else:
                total_count += count
                total_errors += errors

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL DONE! Processed: {total_count} cuts, Errors: {total_errors}, Skipped: {skipped} subsets")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
