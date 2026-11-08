"""Exact + MinHash near-duplicate removal for generated utterances; also length gate by estimated duration.

Exact: lowercase, punctuation stripped. Near: MinHash LSH over word 3-shingles at Jaccard >= `threshold`.
The first occurrence (input order) wins. Writes the kept rows and prints the counts.
"""

import json
import re

import tyro
from datasketch import MinHash, MinHashLSH

from common import write_jsonl_atomic


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def minhash(words: list[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(max(1, len(words) - 2)):
        m.update(" ".join(words[i : i + 3]).encode())
    return m


US_CURRENCY = r"\$|\bUSD\b|\bdollars?\b"  # the prompts ask for a European setting; drop the stragglers


def main(inp: str, out: str, threshold: float = 0.8, num_perm: int = 64, words_per_s: float = 2.6,
         min_s: float = 3.0, max_s: float = 25.0, drop_pattern: str = US_CURRENCY) -> None:  # fmt: skip
    rows = [json.loads(line) for line in open(inp)]
    seen, kept, n_len, n_exact, n_pat = set(), [], 0, 0, 0
    drop = re.compile(drop_pattern, re.IGNORECASE) if drop_pattern else None
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for r in rows:
        if drop and drop.search(r["text"]):
            n_pat += 1
            continue
        words = norm(r["text"]).split()
        est = len(words) / words_per_s
        if not (min_s <= est <= max_s):
            n_len += 1
            continue
        key = " ".join(words)
        if key in seen:
            n_exact += 1
            continue
        seen.add(key)
        m = minhash(words, num_perm)
        if lsh.query(m):
            continue
        lsh.insert(str(len(kept)), m)
        kept.append(r)
    write_jsonl_atomic(out, kept)
    print(
        f"in {len(rows)}: pattern-dropped {n_pat}, length-gated {n_len}, exact dup {n_exact}, "
        f"near dup {len(rows) - n_pat - n_len - n_exact - len(kept)}, kept {len(kept)}"
    )


if __name__ == "__main__":
    tyro.cli(main)
