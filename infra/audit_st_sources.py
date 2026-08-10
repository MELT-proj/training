#!/usr/bin/env python3
"""Audit the ST sources of a training config before spending GPU hours on them.

Speech-translation sources are the easiest thing in this repo to mislabel,
because a wrong tag produces a config that loads, trains, and reports a falling
loss while teaching the model the wrong task. That is not hypothetical: every
``yodas-granary/<Language>/ast`` source shipped in SFT-v1.3.0 was tagged
``src_lang: en, tgt_lang: X`` with the default ``text_field``, when the audio is
X-language and the English translation lives at ``custom.translation_en``. The
direction was inverted *and* the training target was the transcript, so ~64,000 h
of "ST" data was ASR data wearing an ST label.

The invariant it checks is that **the training target must be the translation**.
Corpora disagree about where that lives, and a supervision's ``language`` field
describes the *supervision text*, not the audio:

- CoVoST2 puts the translation in the supervision (``language: de`` for en->de,
  with the English source at ``custom.sentence``). Here ``text_field: text`` is
  correct.
- YODAS Granary puts the *transcript* in the supervision (``language: es`` for
  es->en) and the translation at ``custom.translation_en``. Here ``text_field``
  must point at that custom field.

So the audit reads the supervision language of each sampled cut and requires it
to equal either ``tgt_lang`` (supervision holds the target — plain ``text`` is
right) or ``src_lang`` (supervision holds the transcript — ``text_field`` must
point elsewhere, and must resolve). Matching neither means the tags are wrong.
Whatever the shape, the configured field must resolve to a non-empty value: a
null there makes ``get_text_from_cut`` fall back to the supervision text, which
is exactly how the shipped bug stayed invisible.

Run it where the data is — it reads manifests, not audio.

    python3 infra/audit_st_sources.py \\
        --config        config/train/SFT-v1.3.0.yaml \\
        --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \\
        --sample-cuts   500

Exits non-zero if any source fails, so it can gate a launch script.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path


# Locale codes appear on either side of a pair (``sv-SE_en``), and a language
# tag of ``zh`` should match audio recorded as ``zh-CN``. Compare on the primary
# subtag only.
def _primary(code: str) -> str:
    return str(code).replace("_", "-").split("-")[0].lower()


def _iter_leaves(nodes):
    """Yield leaf source entries from a possibly nested ``input_cfg``."""
    for node in nodes:
        if node.get("type") == "group":
            yield from _iter_leaves(node.get("input_cfg", []))
        else:
            yield node


def _manifest_paths(source_dir: Path) -> list[Path]:
    """Return cut manifests, in either Shar layout.

    An indexed collection stores plain ``cuts.*.jsonl`` beside ``.idx`` byte
    offsets, so it cannot stay compressed. Globbing only the gzipped form
    reports an indexed source as empty.
    """
    paths = sorted(glob.glob(str(source_dir / "cuts.*.jsonl.gz")))
    paths += sorted(glob.glob(str(source_dir / "cuts.*.jsonl")))
    return [Path(p) for p in paths]


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _get_nested(obj, dotted: str):
    current = obj
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def audit_source(source_dir: Path, tags: dict, sample_cuts: int) -> list[str]:
    """Return a list of problem strings for one ST source (empty if clean)."""
    problems: list[str] = []

    text_field = tags.get("text_field") or "text"
    src_lang = tags.get("src_lang")
    tgt_lang = tags.get("tgt_lang")

    manifests = _manifest_paths(source_dir)
    if not manifests:
        problems.append(f"no cut manifests found under {source_dir}")
        return problems

    seen = missing_text = 0
    # Cuts carrying a separate English translation in `custom`. Their
    # supervision is therefore the transcript, which fixes the direction.
    with_translation = 0
    sup_langs: dict[str, int] = {}
    example_missing = None

    for manifest in manifests:
        if seen >= sample_cuts:
            break
        with _open(manifest) as handle:
            for line in handle:
                if seen >= sample_cuts:
                    break
                if not line.strip():
                    continue
                cut = json.loads(line)
                seen += 1

                if text_field != "text":
                    value = _get_nested(cut, text_field)
                    if value is None or not str(value).strip():
                        missing_text += 1
                        example_missing = example_missing or cut.get("id")

                translation = _get_nested(cut, "custom.translation_en")
                if translation is not None and str(translation).strip():
                    with_translation += 1

                sups = cut.get("supervisions") or []
                sup_lang = sups[0].get("language") if sups else None
                if sup_lang:
                    key = _primary(sup_lang)
                    sup_langs[key] = sup_langs.get(key, 0) + 1

    if missing_text:
        problems.append(
            f"{missing_text}/{seen} sampled cuts have no value at "
            f"'{text_field}' (e.g. {example_missing}); these silently fall "
            "back to the supervision text"
        )

    dominant = max(sup_langs, key=sup_langs.get) if sup_langs else None

    if with_translation:
        # A non-null `custom.translation_en` alongside a supervision in another
        # language is decisive: the supervision is the transcript and the
        # English text is the translation, so the pair is <supervision>->en.
        # Inverting the tags produces a config that looks self-consistent on
        # language alone, which is how this shipped unnoticed.
        if dominant and dominant != "en":
            if _primary(tgt_lang or "") != "en":
                problems.append(
                    f"{with_translation}/{seen} sampled cuts carry "
                    f"custom.translation_en beside a '{dominant}' supervision, "
                    f"so this source is {dominant}->en, but it is tagged "
                    f"{src_lang}->{tgt_lang} — the direction is inverted"
                )
            if src_lang and _primary(src_lang) != dominant:
                problems.append(
                    f"src_lang is '{src_lang}' but the supervision text is "
                    f"'{dominant}' on {sup_langs.get(dominant)}/{seen} cuts"
                )
            if text_field != "custom.translation_en":
                problems.append(
                    f"the translation lives at custom.translation_en, but "
                    f"text_field is '{text_field}' — the training target is "
                    "the transcript, not the translation"
                )
    else:
        # No separate translation: the supervision itself must be the target.
        if dominant and tgt_lang and dominant != _primary(tgt_lang):
            problems.append(
                f"the supervision text is '{dominant}' but tgt_lang is "
                f"'{tgt_lang}', and no custom.translation_en is present to "
                "hold the target"
            )
        if text_field != "text":
            problems.append(
                f"text_field points at '{text_field}', but the supervision "
                f"already holds the '{dominant}' target and no separate "
                "translation field was found"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Value for LOCAL_DATASETS_DIR when resolving shar paths. "
        "Defaults to the environment variable.",
    )
    parser.add_argument(
        "--sample-cuts",
        type=int,
        default=500,
        help="Cuts to read per source. The failure modes here are systematic, "
        "so a sample finds them; raise it for a pre-launch gate.",
    )
    args = parser.parse_args()

    import yaml

    root = args.datasets_root or os.environ.get("LOCAL_DATASETS_DIR")
    if root is None:
        print(
            "error: pass --datasets-root or set LOCAL_DATASETS_DIR", file=sys.stderr
        )
        return 2
    root = str(root)

    config = yaml.safe_load(args.config.read_text())
    input_cfg = config["data"]["train_ds"]["input_cfg"]

    st_sources = [
        leaf
        for leaf in _iter_leaves(input_cfg)
        if (leaf.get("tags") or {}).get("task") == "st"
    ]
    if not st_sources:
        print(f"No ST sources in {args.config}; nothing to audit.")
        return 0

    print(f"Auditing {len(st_sources)} ST sources from {args.config}")
    print(f"Reading up to {args.sample_cuts} cuts per source\n")

    failed = 0
    for leaf in st_sources:
        tags = leaf.get("tags") or {}
        shar_path = leaf["shar_path"].replace("${oc.env:LOCAL_DATASETS_DIR}", root)
        label = shar_path.replace(root, "").lstrip("/")

        problems = audit_source(Path(shar_path), tags, args.sample_cuts)
        if problems:
            failed += 1
            print(f"FAIL  {label}")
            for problem in problems:
                print(f"        - {problem}")
        else:
            print(
                f"ok    {label}  "
                f"({tags.get('src_lang')}->{tags.get('tgt_lang')}, "
                f"target={tags.get('text_field')})"
            )

    print()
    if failed:
        print(f"{failed} of {len(st_sources)} ST sources failed the audit.")
        return 1
    print(f"All {len(st_sources)} ST sources passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
