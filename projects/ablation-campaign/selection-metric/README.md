# Selection metric for MA-stage checkpoints

One score used to rank checkpoints from the five-language interface screen
(`plan/01-interface-recipe.md` §3) and the audio-stack crossing
(`plan/03-audio-stack.md` §2). Defined by the PI on 2026-09-23, before the
crossing's results, so the ranking cannot be tuned to them. Lower is better.

## Per-language quantity

Corpus-level **CER** after the standard multilingual normalizer, **clipped at
1.0**: `c_l = min(CER_l, 1.0)`. The clip makes a failed language count as a
failure however much the model hallucinates.

## Groups

The training languages are en, de, es, fr, it. Everything out of domain (OOD) is
scored on the FLEURS **dev** split. The FLEURS-24 set covers the 24 EU
languages plus ru and uk. Every set is a fixed subset drawn once with a fixed seed: **600
utterances per language in-domain** (200 from each held-out corpus) and **200
per language on FLEURS** (see below).

| group | languages | source | weight |
|---|---|---|---|
| **ID**: in-domain | en, de, es, fr, it | dev sets of the corpora trained on, pooled per language | 1.0 |
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

## How it is computed

- **Every number comes from melt-eval, run on the saved checkpoint.** The
  in-training eval history is never used. The in-domain sets are frozen and
  recomputed the same way as FLEURS.
- **Subsets, not full dev sets (PI, 2026-09-23; revised 2026-09-24).** While
  many checkpoints need scoring (the screen now, the crossing next), every
  set is a subset drawn once with a fixed seed (0) and frozen, capped at
  **200 utterances per source** (`max_samples`). Every checkpoint is scored
  on the same utterances. If a source has fewer than 200 dev utterances, it
  uses all of them and says so (FLEURS nl has 171).
- **In-domain:** one frozen melt-eval set per training language, with **200
  utterances from each** held-out split of every corpus trained on that has
  one: `cv22_sidon`, `mls_sidon`, `voxpopuli`, so **600 per language**.
  `yodas-granary` has no held-out split. The three corpora are weighted
  equally, not in proportion to their dev size (the earlier pooled draw was
  dropped). CER is corpus-level over the language's 600.
- **FLEURS:** a new frozen set of 200 per language from the FLEURS dev split,
  all 26 languages. The existing `fleurs24-asr-dev` has only 100 per language,
  so it is not this set.
- **Logs are JSON.** melt-eval runs for this metric pass `--log-format json`.
  Older `.eval` logs are converted with
  `inspect log convert --to json --output-dir <dir>`.
- **Scoring:** a standalone script in this folder reads the JSON logs only
  (no inspect_ai import) and prints the per-language CERs, the group medians
  and the score. It never runs a model.

## Known limitations

- The weights are a choice, not something derived.
- The median of five languages ignores how spread out they are, on purpose.
  Report the per-language values next to it.
- The in-domain and FLEURS sets differ in size, so their eval noise differs.
  A difference between checkpoints smaller than the measured eval noise is
  not a ranking.
- The sets are samples: 600 utterances per language in-domain, 200 on FLEURS.
  The grid measured eval-only noise of 0.002-0.010 WER at 200 per set (board
  2026-09-23), and per-language CER moves at least that much. That figure was
  measured at 200 and is not known for the 600-utterance in-domain sets.
  Full dev sets are deferred, not rejected.
