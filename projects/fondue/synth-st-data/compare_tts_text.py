"""Compare TTS source-text files with the same metrics: python compare_tts_text.py name=path [name=path ...] [--n 3000]."""

import collections
import json
import random
import re
import sys

import spacy


args = [a for a in sys.argv[1:] if "=" in a]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3000
nlp = spacy.load("en_core_web_lg", disable=["parser", "lemmatizer"])
KEEP = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "DATE", "FAC", "NORP"}
DIGIT = re.compile(r"\d")
US = re.compile(r"\b(dollars?|USD|\$|cents?)\b", re.IGNORECASE)
for a in args:
    name, path = a.split("=", 1)
    rows = [json.loads(line) for line in open(path)]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    texts = [r["text"] for r in rows]
    w = [len(t.split()) for t in texts]
    op = collections.Counter(t.split()[0].lower().strip(",") for t in texts)
    op3 = collections.Counter(" ".join(t.split()[:3]).lower() for t in texts)
    ents = [[e.label_ for e in d.ents if e.label_ in KEEP] for d in nlp.pipe(texts, batch_size=256)]
    dense = sum(len(e) >= 2 for e in ents) / len(ents)
    kinds = collections.Counter(k for e in ents for k in e)
    toks = [set(re.findall(r"\w+", t.lower())) for t in texts]
    # cheap near-duplicate proxy: Jaccard >= 0.8 against the previous 200 texts in shuffled order
    near = sum(
        any(len(s & p) / max(1, len(s | p)) >= 0.8 for p in toks[max(0, i - 200) : i]) for i, s in enumerate(toks)
    ) / len(toks)
    print(f"== {name}: {len(rows)} sampled of file")
    print(f"  words mean {sum(w) / len(w):.1f}  p5 {sorted(w)[len(w) // 20]}  p95 {sorted(w)[-len(w) // 20]}")
    print(f"  exact-unique {len({t.lower() for t in texts}) / len(texts):.3f}  near-dup(J>=.8 in window) {near:.3f}")
    print(
        f"  top first-word share {op.most_common(1)[0][1] / len(texts):.3f} {op.most_common(4)}  top 3-gram opening {op3.most_common(1)}"
    )
    print(
        f"  digits {sum(bool(DIGIT.search(t)) for t in texts) / len(texts):.2f}  dollars/USD {sum(bool(US.search(t)) for t in texts) / len(texts):.3f}"
    )
    print(
        f"  entities/utt {sum(len(e) for e in ents) / len(ents):.2f}  dense(>=2) {dense:.2f}  kinds {dict(kinds.most_common(6))}"
    )
    if "domain" in rows[0]:
        by = collections.defaultdict(list)
        for r, e in zip(rows, ents):
            by[r["domain"]].append(len(e))
        print("  entities/utt by domain:", {k: round(sum(v) / len(v), 2) for k, v in sorted(by.items())})
