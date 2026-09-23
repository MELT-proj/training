# Selection metric for MA-stage checkpoints

One score used to rank checkpoints from the five-language interface screen
(`01-interface-recipe.md` §3) and the audio-stack crossing
(`03-audio-stack.md` §2). Defined by the PI on 2026-09-23, before the
crossing's results, so the ranking cannot be tuned to them. Lower is better.

## Per-language quantity

Corpus-level **CER** after the standard multilingual normalizer, **clipped at
1.0**: `c_l = min(CER_l, 1.0)`. The clip makes a failed language count as a
failure however much the model hallucinates.

## Groups

The training languages are en, de, es, fr, it. Everything out of domain (OOD) is
scored on FLEURS. The FLEURS-24 set covers the 24 EU languages plus ru and uk.

| group | languages | source | weight |
|---|---|---|---|
| **ID**: in-domain | en, de, es, fr, it | dev sets of the corpora trained on | 1.0 |
| **OOD-train**: training languages, other domain | en, de, es, fr, it | FLEURS | 1.5 |
| **OOD-related**: related Latin-script languages | pt, ro, nl, da, sv | FLEURS | 0.6 |
| OOD-latin: other Latin-script languages | cs, sk, pl, hr, sl, hu, fi, et, lt, lv, ga, mt | FLEURS | reported only |
| OOD-script: other scripts | bg, el, ru, uk | FLEURS | reported only |

A language is **OOD-related** if it is written in Latin script and belongs
to the same branch as a training language: Romance (pt, ro, related to
es/fr/it) or Germanic (nl, da, sv, related to en/de). The rule is fixed by
genealogy, not by observed scores. Catalan would qualify, but it is not in
FLEURS-24.

## Aggregation

Within a group, take the **median** of `c_l` over its languages. Then:

```
score = (1.0 · med(ID) + 1.5 · med(OOD-train) + 0.6 · med(OOD-related)) / 3.1
```

Always report the three weighted group medians alongside the score, plus the
medians of the two reported-only groups.

## Known limitations

- The weights are a choice, not something derived.
- The median of five languages ignores how spread out they are, on purpose.
  Report the per-language values next to it.
- The in-domain and FLEURS terms come from different eval sizes (200 and 100
  utterances per language). A difference between checkpoints smaller than
  the measured eval noise is not a ranking.

## Open (implementation)

- In-domain CER per language: pool that language's dev sets, or take one
  value per corpus?
- FLEURS dev (100 per language, cheap) or test for the final ranking.
- Where the score is computed: a melt-eval scorer, or a script over the
  in-training eval history plus melt-eval logs.
