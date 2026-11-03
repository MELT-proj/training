"""Pack pool shards (`build_pool.py` output) into work units of about `pairs_per_unit` translations.

A unit is one pool JSONL holding rows of ONE corpus (ids are only unique per corpus),
named `<corpus>-NNN.pool.jsonl` in `--units-dir`. Rows keep shard order. Idempotent:
existing unit files are left alone, so only run it on a finished pool.
"""

import glob
import json
import os
from pathlib import Path

import tyro


def main(pool_dir: str, units_dir: str, corpus: str, pairs_per_unit: int = 250000, src_lang: str = "en") -> None:
    Path(units_dir).mkdir(parents=True, exist_ok=True)
    unit, buf, pairs, made = 0, [], 0, 0

    def flush() -> None:
        nonlocal unit, buf, pairs, made
        if not buf:
            return
        out = Path(units_dir) / f"{corpus}-{unit:03d}.pool.jsonl"
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buf))
            os.replace(tmp, out)
            made += 1
        unit, buf, pairs = unit + 1, [], 0

    for path in sorted(glob.glob(os.path.join(pool_dir, corpus, "pool.*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            r.setdefault("src_lang", src_lang)
            buf.append(r)
            pairs += len(r["targets"])
            if pairs >= pairs_per_unit:
                flush()
    flush()
    print(f"{corpus}: {unit} units ({made} new) in {units_dir}")


if __name__ == "__main__":
    tyro.cli(main)
