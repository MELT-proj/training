"""Manifest overlays over existing audio tars, the SPITE layout.

`overlay_shard` copies one source `cuts.NNNNNN.jsonl` line by line, in order (Shar pairs
cut i with tar sample i, so nothing may be dropped or reordered), lets `annotate` edit
each cut, writes the new manifest plus its index, and symlinks the audio tar (and its
index) relative to the new directory. Audio is never copied. Cuts that `annotate`
leaves without `custom.synth.ok.<tgt>` are dropped at training time by a `cut_filter`
`field` rule (missing: drop).
"""

import gzip
import json
import os
from collections.abc import Callable
from pathlib import Path


SYNTH_VERSION = 1


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return f.readlines()


def link(target: Path, link_path: Path) -> None:
    """Relative symlink; an existing identical link is fine, anything else is an error."""
    rel = os.path.relpath(target, link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and os.readlink(link_path) == rel:
            return
        raise FileExistsError(f"{link_path} exists and is not a link to {rel}")
    link_path.symlink_to(rel)


def set_translation(cut: dict, tgt: str, text: str, synth_fields: dict) -> None:
    """Attach one translation and its per-target synth fields (`{"metricx_xxl": 0.9, ...}`).

    Per-target values live in `custom.synth.<field>.<tgt>` so a `cut_filter` rule can address
    them with a dotted path. `ok.<tgt>` = 1 marks a usable translation.
    """
    custom = cut.setdefault("custom", {})
    custom[f"translation_{tgt}"] = text
    synth = custom.setdefault("synth", {"version": SYNTH_VERSION})
    synth.setdefault("ok", {})[tgt] = 1
    for k, v in synth_fields.items():
        synth.setdefault(k, {})[tgt] = v


def overlay_shard(
    src_cuts: str, src_dir: str, out_dir: str, annotate: Callable[[dict], None], force: bool = False
) -> tuple[str, int]:  # fmt: skip
    from lhotse.indexing import create_jsonl_index

    src_cuts_p, src_dir_p, out_dir_p = Path(src_cuts), Path(src_dir), Path(out_dir)
    num = src_cuts_p.name.split(".")[1]
    out = out_dir_p / f"cuts.{num}.jsonl"
    tar = src_dir_p / f"recording.{num}.tar"
    if not tar.exists():
        raise FileNotFoundError(f"{tar} missing for {src_cuts}")
    out_dir_p.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return num, -1
    n = 0
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in read_lines(src_cuts_p):
            cut = json.loads(line)
            annotate(cut)
            f.write(json.dumps(cut, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, out)
    create_jsonl_index(out)
    link(tar, out_dir_p / tar.name)
    tar_idx = src_dir_p / f"recording.{num}.tar.idx"
    if tar_idx.exists():
        link(tar_idx, out_dir_p / tar_idx.name)
    return num, n
