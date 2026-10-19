#!/usr/bin/env python3
"""Write the job list (TSV for eval_queue.py) of the EU24 test battery.

usage: eu24_jobs.py MODEL [MODEL ...] > jobs.tsv        (MODEL keys below; --label sets sft_label)

Per model: FLEURS-24(+ru/uk) ASR test; for models that translate, X->en and en->X test; and the checkpoint
selection metric's sets (FLEURS-26 dev 200 per language, five in-domain sets). Priority order inside the file:
the biggest jobs first (so they start early), then the selection sets. QOS lists say where a job may run.
"""

import argparse


NYX_SFT = "/mnt/scratch-nyx/giuseppe/melt/outputs-sft"
NYX_SD = "/mnt/scratch-nyx/giuseppe/melt/outputs-sd24"
ART_OUT = "/mnt/scratch-artemis/giuseppe/melt-data/outputs"

# key -> (sft_run metadata, model dir, translates?)
MODELS = {
    "A": (
        "GOLDW-final-A",
        f"{ART_OUT}/GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g",
        False,
    ),
    "SFT-EU24": ("SFT-EU24-from-goldw-bd60-ga4-8g-s1337", f"{NYX_SFT}/SFT-EU24-from-goldw-bd60-ga4-8g-s1337", True),
    "CE24": (
        "SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337",
        f"{NYX_SD}/SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337",
        True,
    ),
    "SD24": (
        "SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337",
        f"{NYX_SD}/SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337",
        True,
    ),
    "SD-SIFT24": (
        "SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337",
        f"{NYX_SD}/SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337",
        True,
    ),
    "A-EuroLLM": (
        "EuroLLM-A",
        f"{ART_OUT}/GOLDW-700asr-whisperlargeF-EuroLLM-17B-Instruct-2512F-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g",
        False,
    ),
    "SD24-EuroLLM": (
        "SD24-azeros-from-EuroLLM-A-mix025-2700h-bd64-8g-s1337",
        f"{NYX_SD}/SD24-azeros-from-EuroLLM-A-mix025-2700h-bd64-8g-s1337",
        True,
    ),
    "SD-SIFT24-EuroLLM": (
        "SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337",
        f"{NYX_SD}/SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337",
        True,
    ),
}
ID_LANGS = ("en", "de", "es", "fr", "it")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    parser.add_argument("--label", default="eu24-repeatfirst")
    parser.add_argument("--skip", nargs="*", default=[], help="frozen sets to leave out")
    args = parser.parse_args()

    rows: list[tuple[int, list[str]]] = []

    def add(priority: int, key: str, frozen: str, qos: str, hours: str) -> None:
        run, model, _ = MODELS[key]
        if frozen not in args.skip:
            rows.append((priority, [run, model, args.label, frozen, qos, hours]))

    for key in args.models:
        runaway_prone = key == "SD-SIFT24-EuroLLM" or key == "SD-SIFT24"
        add(
            0 if runaway_prone else 1,
            key,
            "fleurs24-asr-test",
            "gpu-medium,gpu-long" if runaway_prone else "gpu-medium,gpu-long,gpu-short",
            "08:00:00" if runaway_prone else "05:00:00",
        )
    for key in args.models:
        if MODELS[key][2]:
            add(2 if key != "CE24" else 3, key, "fleurs24-st-xen-test", "gpu-short,gpu-medium,gpu-long", "05:00:00")
    for key in args.models:
        if MODELS[key][2]:
            add(4 if key != "CE24" else 5, key, "fleurs24-st-enx-test", "gpu-short,gpu-medium,gpu-long", "05:00:00")
    for key in args.models:
        add(6, key, "selection-fleurs26-dev200", "gpu-short,gpu-medium,gpu-long", "03:00:00")
    for key in args.models:
        for lang in ID_LANGS:
            add(7, key, f"selection-id-{lang}", "gpu-short,gpu-medium,gpu-long", "01:30:00")

    print("# run\tmodel\tlabel\tset\tqos\ttime")
    for _, row in sorted(rows, key=lambda r: r[0]):
        print("\t".join(row))


if __name__ == "__main__":
    main()
