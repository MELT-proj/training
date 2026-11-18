# Synthetic speech-translation data for Fondue

Code that generates synthetic ST data for Fondue's low-resource languages. The whole collection, Streams A-F, is called
**SpeechBlocks-Synth-v1**. The leaves built here are trained together with the rest of the annealing data. Everything is
written as cuts-only Shar overlays over audio that already exists, plus new Shars for TTS and noisy copies.

> Work in progress: this PR stays open until the generation is finished end to end. Status at the bottom.

## What it makes

| Stream | What it is | Brings | Source audio |
|---|---|---|---|
| A, English speech into 18 languages | en -> 18 targets (bg cs da el fi ga hr hu lt mt pl ro sk uk sv lv et sl), 8 of 18 per segment, entity-dense segments get all 18 | ST | existing English ASR audio (Common Voice 22, GigaSpeech, Granary, VoxPopuli) |
| B, FLEURS read speech into en/de/fr/es/it | FLEURS train transcripts of the 18 LR languages -> en/de/fr/es/it (EuroLLM MT; the FLoRes gold pairs are CC BY-SA and not used), plus the train/test overlap audit | ST | FLEURS (CC BY 4.0) |
| C, native speech of the LR languages into other languages | LR -> en/de/fr/es/it, neighbour pairs (lv-lt, sk-cs, hr-sl, fi-et, da-sv, bg-uk), Y -> LR | ST | native LR and de/fr/es/it ASR audio |
| D, existing ST data, now scored | MetricX scores for ST data that already exists (Granary `ast` X->en, CoVoST2 en->et/sl/lv/sv), so the same quality filter applies | ST | existing Shars (new manifests only) |
| E, synthetic English speech into 18 languages | LLM-written English text -> multi-speaker TTS -> Stream A's MT + QE | ST | synthetic (Kokoro-82M, Apache-2.0) |
| F, noisy and reverberant copies | 30% of the translated A/C/E cuts, one copy each; two copies of the native LR tail; plus a noisy FLEURS-dev eval set | ST (the copies of A, C, E), ASR (the copies of the native tail) | streams above, permissive noise corpora |

MT is `utter-project/EuroLLM-9B-Instruct` (greedy, one candidate per source, instruction-first prompt, no reranking).
QE is `google/metricx-24-hybrid-xxl-v2p6` (Apache-2.0), reference-free; the score is an error value in [0, 25], lower is better. **No threshold is applied here**: the filter is
a `cut_filter` `field` rule at train time on `custom.synth.*`. Only empty or degenerate outputs are dropped. Maltese:
MetricX has no signal for it we trust, so `rt_chrf` (back-translation chrF) is stored for every direction with `mt`.

MetricX runs from Google's own releases: `-bfloat16` (26 GB, fits an A6000) or the fp32 one (52 GB, H100/H200). The bf16 release is
checked against fp32 on a stratified sample before production (see the PR). The code is the pinned `google-research/metricx`
checkout (`PYTHONPATH`), and `use_cache=False` is required with recent transformers.

## Size of SpeechBlocks-Synth-v1

Measured from the written manifests on 2026-11-18 (`python collection_hours.py` writes `hours.json`, `hours_by_corpus.csv`
and `hours_by_direction.csv`). *Audio h* counts every distinct cut once. *Pair-h* counts a cut once per target, which is what
training sees, since every target is its own example. *Kept* = the QE score is at most the per-direction bound of
`thresholds_keep75.csv` (the best 75%; the bound is inclusive, as in `cut_filter`).

| Stream | Corpora | Audio h | Pairs | Pair-h | Kept pairs | Kept pair-h |
|---|---:|---:|---:|---:|---:|---:|
| A | 4 | 2,250.4 | 12,068,114 | 22,332.1 | 9,072,239 | 16,642.6 |
| B | 18 | 160.9 | 246,802 | 804.5 | 185,396 | 603.3 |
| C | 35 | 1,224.2 | 3,561,210 | 6,252.6 | 2,677,050 | 4,748.4 |
| D | 15 | 2,769.3 | 1,598,863 | 2,769.3 | 1,201,283 | 2,024.6 |
| E | 1 | 287.9 | 1,799,072 | 5,181.0 | 1,353,737 | 3,918.9 |
| **A-E** | 73 | 6,692.7 | 19,274,061 | 37,339.6 | 14,489,705 | 27,937.8 |
| F, ST copies | 40 | 1,128.3 | 5,223,434 | 10,126.6 | 3,928,037 | 7,592.0 |
| **A-F** | 113 | 7,821.1 | 24,497,495 | 47,466.2 | 18,417,742 | 35,529.9 |

Stream F also holds the two noisy copies of the native LR tail: 26 ASR leaves, 105,926 cuts, 205.1 h, no ST pairs. Of the
8,026.1 h of audio in the collection, 1,621.3 h are new files (E's 287.9 h of TTS, F's 1,333.4 h of noisy audio); the rest is
existing audio reached through cuts-only overlays. Stream D adds scores to translations that already existed.

## Manifest schema

`custom.translation_<tgt>` holds the text. `custom.synth` holds `version, stream, mt_model, qe_model, prompt_version,
src_lang, src_text_field, n_entities` and, per target, `ok.<tgt>`, `metricx_xxl.<tgt>`, `lid.<tgt>`,
`lid_p.<tgt>`, `len_ratio.<tgt>`, `rt_chrf.<tgt>` (mt directions), `entity_copy_rate.<tgt>`; TTS cuts add `tts`
(model, licence, voice, speed, domain), noisy cuts add `noise` = `{corpus, file_id, licence, category, snr_db, rir_corpus,
rir_id, rir_licence, seed, split, source_cut, source_row, copy}` (see Stream F).

Overlays keep **every** cut of a shard in the original order: Shar pairs cut i with tar sample i. Cuts without a surviving
translation have no `custom.synth.ok.<tgt>` and are dropped at train time (`missing: drop`). Leaves are tagged
`{task: st, src_lang, tgt_lang, text_field}` with `strict_text_field: true`.

## Layout

| File | Role |
|---|---|
| `common.py` | languages, the MT prompt (`PROMPT_VERSION`), ledger and JSONL helpers, degenerate-output check |
| `build_pool.py`, `make_units.py` | select segments per Shar shard (ASR-WER band, spaCy entities, target assignment) and pack them into ~250K-pair units |
| `plan_stream_c.py`, `prep_qe_existing.py`, `plan_fleurs_mt.py`, `fleurs_gold.py` | pools for streams C, D and B, and the B train/test audit |
| `run_mt_qe.py` | one unit: MT (vLLM) then QE (MetricX) in separate processes; chunked, idempotent (`mt/final/dropped.NNNN.jsonl`, `DONE`) |
| `overlay.py`, `write_shar.py` | write the cuts-only overlays from unit outputs |
| `gen_tts_text.py`, `dedup_text.py`, `tts_synth.py` | Stream E: text generation, dedup + length gate, Kokoro synthesis |
| `noise_catalog.py`, `noise_mix.py` | Stream F: licence-filtered noise/RIR catalog (disjoint train/eval), mixer with per-cut provenance |
| `plan_noisy_tail.py`, `noise_all.py` | Stream F: which native-tail cuts get copies (the anneal config's own WER rules), and the driver that runs the mixer on every source |
| `check_noisy.py`, `noisy_fleurs_eval.py` | Stream F: re-creates mixes from their metadata and compares them with the stored audio, writes listening pages; noisy FLEURS-dev eval set |
| `collection_hours.py` | hours of the whole collection from the manifests: audio h, pair-h and kept pair-h per stream, corpus, direction and training leaf |
| `make_mix_spec.py` | the one input of `build_fondue_mix.py synth`: every (corpus, target) leaf with its kept hours and cuts, the QE bound per direction, the noisy tail copies |
| `compare_tts_text.py` | same metrics on two text files (duplicates, openings, entities, currency) |
| `submit_units.sh`, `slurm/` | MN5 submit wrapper (8-node cap, busy-unit check, dry run) and sbatch files for MN5 and artemis |
| `Singularity.synth.def`, `verify_models.py` | MT/QE image (vLLM 0.14.1, lhotse 2) and an offline model check |
| `setup_qwen_venv.sh` | separate venv with vLLM 0.19 for Qwen3.6 text generation |
| `tests/` | unit tests (`common`, `overlay`, `noise`) |

## Environments

- **MT + QE**: `synth.sif`, built from `Singularity.synth.def`. Models must be staged (MN5 has no internet): `hf_cache` with
  EuroLLM-9B-Instruct, the MetricX-24-Hybrid-XXL releases and the `google/mt5-xl` tokenizer, Kokoro-82M, plus `lid.176.bin`.
- **Text generation for TTS**: Qwen3.6-35B-A3B (Apache-2.0) needs `vllm>=0.19`, so it runs in its own venv
  (`setup_qwen_venv.sh`), not in `synth.sif`. It fits on two A6000s with tensor parallel 2 (or one H200); thinking is off.
- MN5 jobs read the submit directory when they *start*: submit from a frozen clone and never edit it while jobs are queued.
- On artemis always pass `--partition`, `--qos` and `--gres`; the h200 node cannot see `/mnt/home`.

## Running

```bash
# pools -> units (inside synth.sif)
python build_pool.py --shar-dir ... --out-dir ... --corpus ... --text-field ...
python make_units.py ...
# MN5, from the frozen clone
bash submit_units.sh --unit-dir .../units --out .../out --dry-run <unit> ...
# artemis (MT on A6000 / QE on H100 or H200): slurm/run_units_artemis.sbatch
# overlays
python write_shar.py --src-dir ... --units ... --unit-dir ... --unit-out ... --out-dir ... --work ...
# tests, inside synth.sif
python -m pytest tests -q -p no:cacheprovider
```

Each script has a docstring with its arguments and invariants. `ruff check` and `ruff format` use `ruff.toml` here
(the root config only covers `melt/`).

## Licence rules

Inputs and tools must allow derivative work with attribution only: CC0 / public domain, CC BY, Apache-2.0. No SA, ND,
NC or Sampling+. Noise: MUSAN (noise, plus PD/BY music only), DEMAND (CC BY 4.0), FSD50K clips with CC0 or CC BY,
OpenSLR 26/28 RIRs (Apache-2.0). `noise_catalog.py --allow` narrows the set (for example to CC BY 4.0 and CC0 only).

**Stream B uses no FLoRes-200 text.** The target side of the FLEURS gold pairs is the FLoRes-200 sentence (CC BY-SA 4.0), which
does not meet the attribution-only rule above, so the gold pairs were dropped (`fleurs_gold.py` is kept for its train/test audit; the
gold join it also writes is not used).
Stream B's targets are EuroLLM translations of the FLEURS transcripts (CC BY 4.0), and none of its manifests carries a
FLoRes field or a share-alike licence marker (checked on all 49,408 cuts of the 18 overlays).

## Stream F: noisy copies

Noisy and reverberant copies are separate Shars with their own 16 kHz FLAC, so training reads them as extra leaves.

- **What is copied.** Streams A, C and E: 30% of the cuts that carry a translation (`custom.synth.ok`), picked by a hash of the
  cut id, one copy each. Every translation and score of the source cut is kept, so the leaf's `cut_filter` rules apply unchanged.
  Native LR tail (ga mt sl hr lv et lt da): every cut of that language's ASR leaves on nyx that passes the annealing config's own
  WER rules (`plan_noisy_tail.py` applies them to the same score arrays `cut_filter.py` reads), two independently seeded copies;
  the new leaf has no `wer_band` tag, so no rule filters it again. VoxPopuli-Mosel exists on MN5 only and is not copied.
- **The mix.** SNR ~ U(0, 20) dB against the (reverberated) speech, an impulse response with probability 1/2, noise from the
  train split of `catalog.json`: a category first (ambience, ambience_misc, crowd, dog, effects, environment, metro_train, music,
  traffic), then a file in it. DEMAND's seven environment groups are one category, because they have 11 recordings in all.
  Everything is seeded from the cut id (and row and copy), so a rerun writes the same audio.
- **Provenance.** `custom.synth.noise` = `{corpus, file_id, licence, category, snr_db, rir_corpus, rir_id, rir_licence, seed, split,
  source_cut, source_row, copy}`; ids are `<source id>-noisy<k>` (`<id>-r<row>-noisy<k>` for the tail, where FLEURS repeats ids).
- **Checks.** `check_noisy.py` checks every manifest entry, then re-creates a stratified sample of mixes from their metadata
  alone and compares them with the stored audio (they agree to one 16-bit step), measures the SNR of the stored audio against the
  recorded one, and compares the labels with the source cut's. It also writes a self-contained `listen.html` with clean and noisy audio.
- **Eval.** `noisy_fleurs_eval.py`: 100 distinct FLEURS validation utterances per language at 5 and 10 dB, noise and RIRs from
  the eval split only (no file or recording environment shared with training).

```bash
python noise_catalog.py                            # catalog.json (FSD50K extracted to noise/fsd50k/audio/{dev,eval} first)
python plan_noisy_tail.py --config Fondue-anneal.yaml --out <out_root>/plan
python noise_all.py --dry-run                      # sizes every source from the manifests
python noise_all.py --workers 40                   # resumable; <name>_noisy/train per source, plus leaves.json
python check_noisy.py --noisy-dir ... --src-dir ... --catalog ... --out ...
python noisy_fleurs_eval.py                        # <out>/snr{5,10}/<locale>/validation
```

## Into Fondue's anneal

`projects/ablation-campaign/fondue/build_fondue_mix.py synth` (training repo) adds the collection to the
rendered annealing config. It reads the text of `Fondue-anneal.yaml`, so it needs no hours cache and cannot drift from what was
deployed. This package supplies its one input:

```bash
python collection_hours.py                 # hours_by_leaf.csv, among others
python make_mix_spec.py                    # mix_spec.json -> fondue/quality/speechblocks_synth_v1.json
python projects/ablation-campaign/fondue/build_fondue_mix.py synth     # writes Fondue-anneal-synth.yaml
```

- **Existing mixture.** Every group keeps its share of the audio relative to the others; all are scaled down together.
- **ST leaves.** One leaf per (corpus, target), one group per direction, weighted by the same alpha/beta over their kept hours. The
  share of the audio drawn is by default the collection's hours' share of all surviving hours (`--share 0.1` to set it).
- **Filter.** One `field` rule per (stream, src, tgt): `custom.synth.metricx_xxl.<tgt>` at most the direction's bound in
  `thresholds_keep75.csv`. A cut with no score for that target is dropped, which is how the untranslated cuts of an overlay leave.
- **Noisy tail copies.** They join the group of their language as corpora under the same alpha rule, and the group keeps its share
  of the audio. A weight counts cuts and the copies' cuts are shorter, so a group that takes copies in has its weight raised
  (1.27 to 1.40x for seven languages; Irish, whose copies are as long as its other cuts, stays). They carry no WER tags: no score
  array exists for a copy.
- **Paths.** `/gpfs/scratch/epor48/synth/shar-indexed/<name>/train` (`--root`); `/gpfs` resolves as is inside the training container.

## Status

See the PR description for the current state; update it as streams finish.
