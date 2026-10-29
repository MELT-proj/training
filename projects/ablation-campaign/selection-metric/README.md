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

## Scoring the 2,100 h crossing

`submit_crossing_eval.sh` (this folder) runs on MN5 and submits the six eval jobs of this
metric (five in-domain sets and FLEURS-26, one GPU each, `-T task_filter=asr --log-format
json -M batch_size=4 -M dtype=bfloat16`, as for the screen) for one or more crossing cells:

    bash <training checkout>/projects/ablation-campaign/selection-metric/submit_crossing_eval.sh \
        [--code ~/training-eval] [--qos acc_debug] [--limit N] [--dry-run] whisper-conformer mms1b-qformer ...
    python3 score.py /gpfs/scratch/epor48/eval-logs/selection-metric/ma2100-<cell> ...

- **Which run.** A cell is a row suffix (`whisper-conformer`). The run scored is the one the
  *current* row of `campaign.yaml` names, which is its composed run name, so the adapter LR,
  the step count and the `qkn` tag (`model.adapter.qk_norm`) are what the row says, and only
  once it has finished: `checkpoint-<max_steps>` and `processor_config.json` must exist. The
  model scored is the run root, the final weights. An unfinished cell is skipped.
- **Which code.** The eval image bakes in `melt` at build time (the v3 image, 2026-09-22,
  has no MoE adapter, an old Conformer and no `qk_norm`) and the MN5 eval runner binds only
  the melt-eval checkout. The script therefore puts the training checkout it was started
  from on `PYTHONPATH` inside the container (`SINGULARITYENV_PYTHONPATH`; the checkout is
  under `$HOME` on GPFS, which the container sees at the same path), so the models load with
  the code that trained them and the image does not have to be rebuilt for every
  training-side change. This is the eval-side twin of the training launcher, which runs from
  the bind-mounted checkout (`docs/hpc_runbook.md`, "Mental model"); the melt-eval code
  already runs from the checkout the jobs are submitted from. A checkpoint trained without
  `qk_norm` has no such key in its config, which means off. `--code DIR` names the training
  checkout (default: the one the script sits in). For the crossing it is `~/training-eval`
  on MN5, a frozen clone of `main` at the merge of #149. Both checkouts are read when a job
  *starts*, hours after submission on acc_ehpc, so neither `~/training-eval` nor `~/eval`
  may be switched, pulled or edited while jobs are pending.
- **Logs and ledger.** JSON logs go to `/gpfs/scratch/epor48/eval-logs/selection-metric/ma2100-<cell>/<set>/`;
  each submission is appended to `~/eval/frozen-sets/selection-metric/eval-jobs.tsv` with the
  code commit. A set that already has a JSON log, or whose job is queued or running, is
  skipped, never overwritten or doubled. `--limit N` writes to a separate `-smoke` tree that
  `score.py` is never pointed at.
- **Cost.** The screen measured 0.8-0.9 GPU-h per Whisper checkpoint and 1.4-1.6 for
  w2v-BERT (FLEURS 35-70 min, each in-domain set 3-6 min); sixteen cells should take
  roughly 20-25 GPU-h. The MMS-1b and mHuBERT cells are not measured.
- **Reading the scores.** The grid is not uniform (adapter LR and `qk_norm` differ by cell,
  `campaign.yaml` header): list each cell's LR and `qk_norm` next to its score.
