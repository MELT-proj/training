"""Stream B: pools and units that translate the FLEURS train transcripts of the 18 low-resource languages into Y.

Replaces the FLoRes gold join (its target text is CC BY-SA). FLEURS audio and transcripts are CC BY 4.0;
the translations are EuroLLM-9B-Instruct output, scored by MetricX-24-Hybrid-XXL like every other stream. FLEURS cut
ids are only unique per locale, so each language is its own corpus `fleurs_<lang>` (one unit each).
Run inside synth.sif; the train/test audit stays in `fleurs_gold.py`.
"""

from pathlib import Path

import tyro

from build_pool import main as build_pool
from fleurs_gold import FLEURS_DIR, LR_LANGS, Y_LANGS
from make_units import main as make_units


def main(
    shar_root: str = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar",
    pool_dir: str = "/mnt/scratch-nyx/giuseppe/synth/pool",
    units_dir: str = "/mnt/scratch-nyx/giuseppe/synth/units",
    workers: int = 8,
) -> None:
    for lang in LR_LANGS:
        corpus = f"fleurs_{lang}"
        build_pool(
            shar_dir=str(Path(shar_root) / "fleurs" / FLEURS_DIR[lang] / "train"),
            out_dir=pool_dir,
            corpus=corpus,
            text_field="supervisions.0.text",
            min_dur=1.0,
            src_lang=lang,
            targets=Y_LANGS,
            workers=workers,
        )
        make_units(pool_dir, units_dir, corpus, pairs_per_unit=10**9, src_lang=lang)


if __name__ == "__main__":
    tyro.cli(main)
