"""Stream F, eval: a noisy FLEURS dev set built from the EVAL split of the noise catalog.

For every language, `n` distinct utterances of FLEURS `validation` (the smallest hashes, so the choice is fixed and
independent of shard order; repeated ids, from the interrupted FLEURS re-ingest, are taken once) are mixed at each
fixed SNR (5 and 10 dB). The variants share the utterances, the noise file, the noise offset and the RIR (the draw
does not depend on the SNR), so the two SNRs differ in level only. Noise and RIRs come from the eval split, which shares
no file or recording environment with the one the training copies use.

Output: `<out>/snr<k>/<locale>/validation/` (indexed Shar with its own 16 kHz FLAC) and `noise_stats.json` per locale.
This only writes new files under `<out>`; whether and where melt-eval takes them is the eval owner's call.
Run inside synth.sif.
"""

import json
from pathlib import Path

import numpy as np
import tyro

from noise_mix import SALT, h, list_shards, open_text
from noise_mix import main as noise_mix


FLEURS = "/mnt/scratch-nyx/giuseppe/melt/melt-data/shar/fleurs"
CATALOG = "/mnt/scratch-nyx/giuseppe/synth/noise/catalog.json"
OUT = "/mnt/scratch-nyx/giuseppe/synth/eval-noisy/fleurs-dev"
# Fondue's languages: the 24 EU ones, Ukrainian and Russian (FLEURS locale codes)
LOCALES = [
    "bg_bg",
    "cs_cz",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fi_fi",
    "fr_fr",
    "ga_ie",
    "hr_hr",
    "hu_hu",
    "it_it",
    "lt_lt",
    "lv_lv",
    "mt_mt",
    "nl_nl",
    "pl_pl",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sk_sk",
    "sl_si",
    "sv_se",
    "uk_ua",
]


def choose(shar_dir: str, n: int) -> np.ndarray:
    """Bool mask over the rows of a Shar dir: the first occurrence of each of the `n` smallest-hash distinct ids."""
    ids = []
    for cuts, _ in list_shards(shar_dir):
        with open_text(cuts) as f:
            ids += [json.loads(line)["id"] for line in f]
    first: dict[str, int] = {}
    for row, cid in enumerate(ids):
        first.setdefault(cid, row)
    ranked = sorted(first, key=lambda c: h(c + SALT + "evalsel"))[:n]
    mask = np.zeros(len(ids), dtype=bool)
    mask[[first[c] for c in ranked]] = True
    return mask


def main(
    out: str = OUT,
    catalog: str = CATALOG,
    fleurs: str = FLEURS,
    locales: tuple[str, ...] = tuple(LOCALES),
    snrs: tuple[float, ...] = (5.0, 10.0),
    n: int = 100,
    workers: int = 4,
) -> None:
    rows = []
    for loc in locales:
        src = f"{fleurs}/{loc}/validation"
        mask_file = Path(out) / "masks" / f"{loc}.npy"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_file, choose(src, n))
        for snr in snrs:
            dest = f"{out}/snr{int(snr)}/{loc}"
            if (Path(dest) / "noise_stats.json").exists():
                continue
            rep = noise_mix(
                src_dir=src, out=dest, catalog=catalog, split="eval", keep_mask=str(mask_file), snr_db=snr,
                tag=f"snr{int(snr)}", stream="noisy_eval", workers=workers, subdir="validation",
            )  # fmt: skip
            rows.append((loc, snr, rep["noisy_cuts"], rep["hours"]))
    for loc, snr, cuts, hours in rows:
        print(f"{loc:6} snr {snr:>4}  {cuts:4d} utterances  {hours:.3f} h")


if __name__ == "__main__":
    tyro.cli(main)
