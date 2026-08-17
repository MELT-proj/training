# Verifying a training config before you launch

`infra/check_training_config.py` reads a training config and checks that the numbers in its
`data:` section still describe the data it points at. Run it before scheduling a job.

The numbers it checks are derived offline and pasted in, so they go stale quietly. Nothing in the
training stack re-derives them, and most of the ways they can be wrong do not raise:

| what goes wrong | what training does about it |
| --- | --- |
| `total_hours` is stale | feeds `max_steps`, so warmup and decay silently rescale |
| `bucket_duration_bins` is stale or malformed | handed straight to lhotse, never validated |
| no `name:` on validation sources | every validation set pools into one `eval_loss` |
| mixture `weight` values are wrong | only all-or-none within a level is enforced |

```bash
python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml
```

Exit status is 0 when everything checks out, 1 when a check fails, 2 when the script could not run.
`--strict` makes warnings count as failures too, which is what you want in CI.

## Where to run it

Where the data is, like `compute_mix_weights.py`. It needs **PyYAML and numpy only** — no lhotse,
no torch, no joblib. On nyx:

```bash
/mnt/scratch-nyx/giuseppe/venvs/lhotse2/bin/python infra/check_training_config.py \
    --config config/train/SFT-v1.4.0.yaml \
    --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar
```

`--datasets-root` defaults to `$LOCAL_DATASETS_DIR`. To check structure alone, with no data
access and no root at all:

```bash
python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml --offline
```

## The measurement cache

Checking hours, cut counts and bins means knowing what is actually in the manifests. A full pass
over the SFT mixture is ~22k shards, so measurements are cached in
`infra/.config_check_cache.json` (gitignored: absolute paths, machine-specific).

Per source it holds unfiltered hours, the cut count, the shard count, the newest manifest mtime,
and a duration histogram quantised to 0.01 s. The histogram is why one pass answers bin questions
for **any** duration filter and **any** `num_buckets` afterwards: it reproduces the exact
estimator to within one 0.01 s step, which is below the two decimal places configs are written at.

The cache is seeded from `compute_mix_weights.py`'s `infra/.mix_weights_hours.json` when that
exists. Both tools sum the same unfiltered top-level `duration` over the same manifests, so hours
transfer exactly. That file records no cut counts and no histograms, though, so a seeded source can
answer the weight and `total_hours` checks but not `total_cuts` or the bins — those show as
`skip` until measured.

By default the script fills only small gaps (`--max-auto-shards 200`, enough for a validation
split) and tells you what it left. To measure the rest:

```bash
python3 infra/check_training_config.py --config config/train/SFT-v1.4.0.yaml --measure --jobs 32
```

That is resumable — the cache is written as each source completes, so an interrupted run picks up
where it stopped. Use `--force-measure` to distrust the cache entirely, and `--no-probe` to skip
the one-cut read per source that check T4 needs.

A source whose manifests were rewritten since it was measured is detected by mtime and re-measured
automatically. That matters after a `pnc_text` or `num_tokens` backfill, which changes effective
hours.

## What it checks

Failures — these are the four things the script exists for:

| id | check |
| --- | --- |
| `W1` | within a level, either every entry sets `weight` or none does |
| `W2` | each level's weights sum to 1, and none is zero or negative |
| `W3` | every `p_l`/`p_c` matches the two-tier policy replayed over measured hours at the config's own alpha/beta |
| `W4` | each group is exactly one language entry, and its tags agree with its members' |
| `N1` | every `validation_ds` source declares a `name` |
| `N2` | naming is not partial (the loader raises on that) |
| `H1`/`H2` | `train_ds.total_hours` / `total_cuts` match the measured mixture |
| `H3` | `validation_ds` totals are consistent |
| `B1` | `dynamic_bucketing` declares `num_buckets` |
| `B2` | `bucket_duration_bins` has `num_buckets - 1` values, ascends, and lies inside the duration filter |
| `B3` | the bin values match this mixture's measured duration distribution |

Warnings — silent-failure classes found while building this, kept advisory so an unusual but
deliberate config still exits 0:

| id | check |
| --- | --- |
| `T1` | no group tag overrides a differing leaf tag (the group wins at load time) |
| `T2` | ASR sources tag `lang`; ST sources tag `src_lang` and `tgt_lang` |
| `T3` | locale codes are folded by `LOCALE_ALIASES` |
| `T4` | `text_field` resolves on a real cut, and an ST source is pointed at a translation |
| `T5` | leaves of one corpus and task agree on `text_field` |
| `D1` | every `shar_path` exists and holds cut manifests |
| `D2` | `.idx` sidecars are complete and newer than their manifests |
| `D3` | no source appears in both `train_ds` and `validation_ds` |
| `E1` | `force_estimate` can actually measure this `input_cfg` (it does not recurse into groups) |
| `E2` | `total_cuts` is usable by the `batch_size` code path |
| `E3` | steps per epoch is derivable at all |
| `C1` | `strict_text_field` reaches the eval path (it is not inherited) |
| `C2` | `per_device_eval_batch_size` is not `-1` while eval is on |
| `C3` | `max_duration` fits the encoder's audio window |
| `C4` | the top bucket covers the long tail of durations |
| `C5` | bins were measured for this mixture rather than copied from another config |

Restrict the report with `--only`, repeatable: `--only weights --only bins`.

### Two things it deliberately does not do

**It never edits a config.** Mixture changes belong on the path in
[mixture_weights.md](mixture_weights.md): edit the *template*, re-run `compute_mix_weights.py`,
re-emit. Anything written into an emitted config's `train_ds.input_cfg` is lost on the next
regeneration, so the script prints the command and the key to change instead of doing it.

**It does not hardcode which text field a corpus uses.** An earlier draft carried that table,
taken from the 2026-08-11 collection audit, and it was already wrong — voxpopuli and MLS Polish
have since been backfilled with `custom.pnc_text`, so the table produced five false positives.
T4 reads one real cut per source instead. What the manifests cannot tell you — whether two leaves
of one corpus *should* agree, and whether an ST source is pointed at a translation rather than a
transcript — is checked structurally in T5 and T4.

## Reading the output

Checks are grouped by section with `ok` / `warn` / `FAIL` / `skip`, then a `measured` block with
the numbers it derived, then a `WHAT TO DO` block that repeats only what needs action — each item
with a `config/train/X.yaml:1638` location and either a runnable command or the literal YAML to
paste.

`skip` always names its reason. Usually it is a missing measurement, in which case the script also
prints the `--measure` command that fills it.

Two things worth knowing when reading the numbers:

- **Duplicated sources are counted twice, on purpose.** The 22 `yodas-granary/*/ast` leaves appear
  once as `task: asr` and once as `task: st`, because the mixture draws from each twice. Both
  `total_hours` and `total_cuts` include them twice, and the script reports the distinct-audio
  figure beside the total so the difference is legible rather than looking like a 74k h error.
- **`total_hours` and `total_cuts` are checked unfiltered**, matching how they were derived and what
  `compute_mix_weights.py` writes. The figures after `min_duration`/`max_duration` are printed
  beside them, so a large gap is visible.

## JSON output

`--json report.json` writes every check with its status, findings and suggested fixes. Useful for
CI or for diffing two configs.

## Tests

```bash
singularity exec --bind /mnt/scratch-nyx,/mnt/scratch-artemis \
    /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif \
    bash -c 'source /workspace/venv/bin/activate && \
      PYTHONPATH=/mnt/scratch-nyx/giuseppe/container-extras \
      python -m pytest tests/test_check_training_config.py -q'
```

The container venv has no pytest, hence the host-side overlay on `PYTHONPATH`.

The checker restates rules that live in the training loader — the tag-merge order of nested groups,
the all-or-none weight rule, the validation naming rule. Where the real function is importable the
tests assert the two **agree**, rather than only testing the copy, because a mirror that drifts is
worse than no check at all.
