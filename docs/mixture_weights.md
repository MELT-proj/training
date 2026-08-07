# Computing mixture weights for a training config

Run this whenever the set of source datasets changes — adding a corpus, dropping
one, or building an ablation config. The output is a training config whose
`train_ds.input_cfg` carries per-source sampling weights.

## What the weights are

Two-tier language/corpus balancing, from Section 3.3.1 of
[arXiv:2509.14128](https://arxiv.org/pdf/2509.14128) (Canary-1B-v2). Corpora are
balanced *within* a language first, then languages against each other, so a small
corpus is not swamped before cross-language balancing runs:

1. **Corpus within a language.** `w_c = (n(c) / N_l) ** alpha`, normalised to `p_c`.
2. **Language across the mix.** `w_l = (n(l) / N_total) ** beta`, normalised to `p_l`.
3. **Final.** `p_cl = p_l * p_c`.

`n(.)` is **hours of audio**, not utterance counts. `alpha = beta = 0.5` is the
paper's pre-training setting and the script default. Lower values flatten harder;
the paper uses `alpha = 0.2` for one fine-tuning stage.

Every translation **direction** is its own language entry: ASR German is `de`,
while en→de and de→en are `en-de` and `de-en`. This falls out of the config tags
(`lang` for ASR, `src_lang`/`tgt_lang` for ST) — you don't declare it anywhere.

## Where to run it

The script reads every `cuts.*.jsonl.gz` in every source, so run it where the
data is, not on a login node with the data behind a slow mount.

| host | path to the collection |
| --- | --- |
| Artemis / nyx | `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar` |
| MN5 | `/gpfs/projects/epor48/melt-data/shar` |

Only `PyYAML` is needed — no lhotse, no torch, so the system `python3` is fine
and there is no need for the container.

## The command

```bash
python3 infra/compute_mix_weights.py \
    --config       config/train/SFT-v1.2.7.yaml \
    --datasets-root /mnt/scratch-nyx/giuseppe/melt/melt-data/shar \
    --cache        hours.json \
    --emit-config  config/train/SFT-v1.3.0.yaml \
    --exp-name     SFT-v1.3.0
```

`--config` is both the source of truth for which datasets are in the mix *and*
the template for the emitted config: everything outside `train_ds.input_cfg` is
copied through byte for byte, comments included. Only the input block and
`train_ds.total_hours` change.

Other outputs, if you want them separately:

- `--emit-nemo weights.yaml` — just the nested `input_cfg` block. This is the
  schema NeMo Speech reads, so the same file can drive a NeMo run over the same
  shards (relevant for Smurf).
- `--emit-yaml flat.yaml` — a flat `shar_path: p_cl` mapping, for inspection.

## Timing, and the cache

A full pass over the current collection is **22k shards and about two hours**,
because it decompresses and parses every manifest. It is I/O-bound, so the
`--jobs` default is fine and raising it past ~32 does not help.

`--cache` is written **as each source finishes**, so an interrupted run resumes
rather than starting over. Point it at the same file next time and only new or
changed sources get measured. Keep the cache — it turns a re-run after adding one
corpus into a minute of work.

For a quick sanity check rather than a config you will train on:

```bash
--sample-shards 4      # read 4 shards per source and scale by the shard ratio
```

Sources with 4 or fewer shards are read in full either way, so small corpora stay
exact. Do not ship a config built this way: extrapolation error on the large
sources was ~8% on the current collection.

## Adding a dataset

1. Add the `- type: lhotse_shar` entry to the **template** config's
   `train_ds.input_cfg`, with its `tags` (`task`, `lang` or `src_lang`/`tgt_lang`,
   and `text_field` if the text is not at `text`). Do **not** add a `weight` —
   the script computes it.
2. Re-run the command above with the existing `--cache`.
3. Check the summary line and the folding report (below) before committing.

### If it carries a locale code

Codes like `de_de`, `pt_br`, `zh-TW` are folded onto their language by
`LOCALE_ALIASES` in the script. A locale that gets its own language entry
collects its own language-level share under beta, so one language split across
two tag spellings is upsampled purely for being spelled two ways — `fleurs/it_it`
is 9 hours and was drawing a full language share against `mls_sidon/italian`'s
247 h, a 45x boost.

The table is **exhaustive, not derived from a rule**: a "strip everything after
the first `-` or `_`" heuristic also mangles codes whose suffix is not a region,
and a silent mis-split quietly changes the sampling distribution. **A new locale
code means adding a row to `LOCALE_ALIASES`.** If you forget, the code becomes
its own language entry and gets over-weighted, which the run will not tell you.

Nothing is lost: every source carries the original code as `region_code`, and ST
sources carry `src_region_code` / `tgt_region_code`, because either side may hold
a locale (`covost2/sv-SE_en` on the source, `en_sv-SE` on the target).

## Reading the output

```
Locale codes folded onto their language (the original is kept per-cut as *region_code):
    de     <- de_de
    zh     <- zh-CN, zh-HK, zh-TW

alpha=0.5  beta=0.5  71 language entries  245,099.2 h total

     p_cl  nat.share   boost       hours lang       source
 0.093915   0.418022    0.22x    102457.0 en         .../yodas-granary/English/asr_only
 0.014214   0.002707    5.25x       663.6 en-pl      .../yodas-granary/Polish/ast
```

`nat.share` is the source's share of total hours; `boost` is `p_cl / nat.share`.
Large corpora should come out below 1x and small ones above — that is the policy
working. Worth checking:

- **Language entry count.** If it jumped, a locale code is probably unfolded.
- **Very high boosts on tiny sources.** A source with under ~10 hours can pull a
  400x boost. That is arithmetic, not a bug, but decide whether you want it in
  the mix at all.
- **Sources reported as 0 h.** These are warned about explicitly and mean the
  glob found no manifests — usually a path typo or an unsynced dataset.

## How the weights take effect

The emitted config nests one `type: group` per language entry, with `p_l` on the
group and `p_c` on each corpus inside it:

```yaml
input_cfg:
  # en: 151,856.2 h across 10 corpora
  - type: group
    weight: 0.21914747          # p_l
    tags: {task: asr, lang: en}
    input_cfg:
      - type: lhotse_shar
        shar_path: ${oc.env:LOCAL_DATASETS_DIR}/yodas-granary/English/asr_only
        weight: 0.42854849      # p_c  ->  sampled at p_l * p_c
        tags: {task: asr, lang: en, region_code: en}
```

The two levels are muxed separately, so the effective probability is the product.
**Do not pre-multiply them**, and do not set NeMo's `reweight_temperature`
alongside this file — alpha and beta are already baked in, and it would raise the
weights to a further power.

Paths keep the `${oc.env:LOCAL_DATASETS_DIR}` placeholder, so the same config
works on Artemis, nyx, and MN5 without editing.

Two constraints the loader enforces:

- Within a level, **either every entry sets `weight` or none does**. Mixing them
  is rejected: an explicit weight is a share of the level (around 1) while an
  automatic one is a raw cut count (in the millions), and muxing normalises both
  onto one scale, starving the explicitly weighted sources.
- Sources are repeated **before** being muxed. `CutSet.mux` drains a source and
  then drops it, so muxing finite sources and repeating the combination delivers
  100% of every corpus per cycle and the mixture tracks corpus size rather than
  the weights. This is handled in `read_cutset_from_config`; it matters if you
  ever call `CutSet.mux` yourself.

## Checking a config before training on it

```python
import yaml
d = yaml.safe_load(open("config/train/SFT-v1.3.0.yaml"))
ic = d["data"]["train_ds"]["input_cfg"]
assert abs(sum(g["weight"] for g in ic) - 1) < 1e-6
for g in ic:
    assert abs(sum(c["weight"] for c in g["input_cfg"]) - 1) < 1e-6
print(len(ic), "language entries",
      sum(len(g["input_cfg"]) for g in ic), "corpora")
```

## Caveats

- `train_ds.total_hours` is updated from the measurement; it drives the
  steps-per-epoch estimate when `batch_size` is null. **`total_cuts` is not
  updated** and will still hold the template's value.
- The emitted config is regenerated wholesale. Hand edits inside
  `train_ds.input_cfg` are lost on the next run — make them in the template.
- Coverage caveat: with the current loader a nominal epoch does not correspond to
  100% of the data, so "epochs" here are nominal. See issue #52.
