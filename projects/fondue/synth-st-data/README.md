# Synthetic speech-translation data for Fondue (arm B)

Code that generates synthetic ST data for Fondue's low-resource languages. It feeds **arm B**, an A/B anneal started
from Fondue's pre-anneal checkpoint and identical to the normal anneal ("arm A") except for the extra leaves built here.
Everything is written as cuts-only Shar overlays over audio that already exists, plus new Shars for TTS and noisy copies.

> Work in progress: this PR stays open until the generation is finished end to end. Status at the bottom.

## What it makes

| Stream | Content | Source audio |
|---|---|---|
| A | en -> 18 targets (bg cs da el fi ga hr hu lt mt pl ro sk uk sv lv et sl), 8 of 18 per segment, entity-dense segments get all 18 | existing English ASR audio |
| B | gold LR <-> Y (en de fr es it) from FLEURS n-way parallelism, plus the train/test overlap audit | FLEURS |
| C | LR -> en/de/fr/es/it, neighbour pairs (lv-lt, sk-cs, hr-sl, fi-et, da-sv, bg-uk), Y -> LR | native LR and de/fr/es/it ASR audio |
| D | XCOMET scores for ST data that already exists (Granary `ast` X->en, CoVoST2 en->et/sl/lv/sv) | existing Shars (new manifests only) |
| E | LLM-written English text -> multi-speaker TTS -> Stream A's MT + QE | synthetic (Kokoro-82M, Apache-2.0) |
| F | noisy / reverberant copies (~30% of A/C/E, native LR tail), noisy FLEURS-dev eval | streams above, permissive noise corpora |

MT is `utter-project/EuroLLM-9B-Instruct` (greedy, one candidate per source, instruction-first prompt, no reranking).
QE is `Unbabel/XCOMET-XXL`, reference-free, score and error spans stored. **No threshold is applied here**: the filter is
a `cut_filter` `field` rule at train time on `custom.synth.*`. Only empty or degenerate outputs are dropped. Maltese:
XCOMET is unreliable, so `rt_chrf` (back-translation chrF) is stored for every direction with `mt`.

XCOMET-XXL runs in **fp32 on H100/H200 only** (43 GB; batch 16 with an OOM retry on a 64 GB card). Do not run it in bf16.

## Manifest schema

`custom.translation_<tgt>` holds the text. `custom.synth` holds `version, stream, mt_model, qe_model, prompt_version,
src_lang, src_text_field, n_entities` and, per target, `ok.<tgt>`, `xcomet_xxl.<tgt>`, `xcomet_spans.<tgt>`, `lid.<tgt>`,
`lid_p.<tgt>`, `len_ratio.<tgt>`, `rt_chrf.<tgt>` (mt directions), `entity_copy_rate.<tgt>`; TTS cuts add `tts`
(model, licence, voice, speed, domain), noisy cuts add `noise` (corpus, file, licence, category, SNR, RIR, seed, split, source cut).

Overlays keep **every** cut of a shard in the original order: Shar pairs cut i with tar sample i. Cuts without a surviving
translation have no `custom.synth.ok.<tgt>` and are dropped at train time (`missing: drop`). Leaves are tagged
`{task: st, src_lang, tgt_lang, text_field}` with `strict_text_field: true`.

## Layout

| File | Role |
|---|---|
| `common.py` | languages, the MT prompt (`PROMPT_VERSION`), ledger and JSONL helpers, degenerate-output check |
| `build_pool.py`, `make_units.py` | select segments per Shar shard (ASR-WER band, spaCy entities, target assignment) and pack them into ~250K-pair units |
| `plan_stream_c.py`, `prep_qe_existing.py`, `fleurs_gold.py` | pools for streams C and D, gold join and audit for B |
| `run_mt_qe.py` | one unit: MT (vLLM) then QE (XCOMET) in separate processes; chunked, idempotent (`mt/final/dropped.NNNN.jsonl`, `DONE`) |
| `overlay.py`, `write_shar.py` | write the cuts-only overlays from unit outputs |
| `gen_tts_text.py`, `dedup_text.py`, `tts_synth.py` | Stream E: text generation, dedup + length gate, Kokoro synthesis |
| `noise_catalog.py`, `noise_mix.py` | Stream F: licence-filtered noise/RIR catalog (disjoint train/eval), mixer with per-cut provenance |
| `compare_tts_text.py` | same metrics on two text files (duplicates, openings, entities, currency) |
| `submit_units.sh`, `slurm/` | MN5 submit wrapper (8-node cap, busy-unit check, dry run) and sbatch files for MN5 and artemis |
| `Singularity.synth.def`, `verify_models.py` | MT/QE image (vLLM 0.14.1, comet 2.2.7, lhotse 2) and an offline model check |
| `setup_qwen_venv.sh` | separate venv with vLLM 0.19 for Qwen3.6 text generation |
| `tests/` | unit tests (`common`, `overlay`, `noise`) |

## Environments

- **MT + QE**: `synth.sif`, built from `Singularity.synth.def`. Models must be staged (MN5 has no internet): `hf_cache` with
  EuroLLM-9B-Instruct, XCOMET-XXL, the xlm-roberta-xxl config, Kokoro-82M, plus `lid.176.bin`.
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

**Open exception, Stream B.** The target-side text of the FLEURS-gold pairs is the FLoRes-200 sentence (CC BY-SA 4.0), and
FLoRes also supplies the sentence ids; the audio and source transcripts are FLEURS (CC BY 4.0). Each such cut records
`CC-BY-SA-4.0 (FLoRes-200) / CC-BY-4.0 (FLEURS)` in `custom.synth.licence`. This does not meet the attribution-only rule
above; whether to keep Stream B is a decision for the project owner (dropping it removes only the gold pairs).

## Status

See the PR description for the current state; update it as streams finish.
