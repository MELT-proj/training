# Evaluation protocol: benchmarks, metrics, baselines

This is the cold-start document for the evaluation side of the paper ("MELT", deadline 2027-01-04). It says which
sets and metrics the paper's tables use, which models are compared, how a number gets from a GPU into the master
table, and what is not comparable with what. It does not hold results: those live in
`projects/ablation-campaign/charts/results_master.csv` (rules: `charts/PROTOCOL.md`). Status as of 2026-10-18.

## 1. The rules on one screen

1. Every measured number a chart or table quotes lives in the master table and is written only through
   `charts/master.py` (or the `ingest_*.py` scripts that call it). Read `charts/PROTOCOL.md` before the first write.
2. Test sets are the numbers the paper quotes. Dev sets are an internal check. Never mix them in one comparison.
3. WER and CER are percent in the master. Translation is scored lowercased (`chrfpp-lc`) as the primary metric;
   the cased `chrfpp` columns exist but are not the headline.
4. A model that does not cover a language or direction gets an empty cell. Never fill it with a made-up score.
5. Baselines run with the prompt their model card documents (format `native`); our checkpoints run with the prompt
   they were trained on (format `repeatfirst`). Same set names, different `fmt`, so the two regimes sit side by side.
6. GPU work goes through `sbatch` on artemis only. Logs are JSON (`--log-format json`).

## 2. Benchmarks

| Set (master name) | Task | Content | Size |
|---|---|---|---|
| `fleurs24-asr-test` | ASR | FLEURS test, the 24 EU languages | 19,463 clips, 63 h |
| `fleurs24-st-xen-test` | speech translation X→en | FLEURS test, 23 source languages | 18,816 clips |
| `fleurs24-st-enx-test` | speech translation en→X | FLEURS test, 23 target languages | 14,457 clips |
| `librispeech-clean`, `librispeech-other` | English ASR | LibriSpeech test-clean (2,620 clips) and test-other (2,939) | 5,559 clips |
| `voxpopuli-test` | multilingual ASR | VoxPopuli test, 16 languages (cs de en es et fi fr hr hu it lt nl pl ro sk sl) | 16,529 clips, 47 h |

The 24 languages: bg cs da de el en es et fi fr ga hr hu it lt lv mt nl pl pt ro sk sl sv. Translation covers the
23 pairs with English. VoxPopuli is usable at scale only for de en es fr it (et and lt have 51 and 41 test clips).
Tiers (high / mid / low by transcribed hours) are in `plan/data/`.

Dev counterparts (internal): `fleurs24-asr-dev`, `fleurs24-st-xen-dev`, `fleurs24-st-enx-dev`, 100 clips per
language or direction; plus the selection-metric sets `selection-id` and `selection-fleurs26-dev200` (§4).

All sets are melt-eval **frozen sets**: a manifest of audio locators (zero-copy, no audio duplicated), identical clips
for every model. They live on artemis under `/mnt/scratch-artemis/giuseppe/melt-data/eval-sets/`. The registry of
sets in the master is `charts/results_sets.csv`.

## 3. Metrics

| Metric | Definition | Where |
|---|---|---|
| WER, CER | melt-eval `asr_scorer`: the same text normalizer (`basic`) applied to reference and hypothesis, corpus-level per language (total errors over total words), percent. Aggregates average languages equally. | ASR sets |
| chrF++ | sacrebleu chrF with word order 2. `chrfpp-lc` is computed over lowercased text and is the **primary** translation metric; `chrfpp` is cased. Aggregates are macro-means over directions. | ST sets |
| spBLEU | BLEU on SentencePiece tokens (FLORES-200 model) | ST sets |
| selection score | the campaign's ASR-only internal model-selection metric (`selection-metric/README.md`, lower is better) | internal |

Traps:
- melt-eval's native `corpus_chrf` is already chrF++ and is **cased**. Our earlier self-distillation tables used plain
  chrF, lowercased. They are not comparable with `chrfpp-lc` numbers. Models that write translations without capitals
  differ by 4–5 points between cased and lowercased scoring.
- A mean over languages is only comparable between models that cover the same languages. Compare on the shared subset
  or say which languages each mean covers.
- WER on the 5 trained languages versus the other 19 behave very differently (FLEURS train data is in our training
  mixes, so FLEURS dev/test are closer to in-domain for the mid and low tiers).

## 4. Models compared

**Off-the-shelf baselines** (`kind=baseline`, fmt `native`), 2026-10-16 decision:

| Model | Covers | Prompt (from the model card) |
|---|---|---|
| `google/gemma-4-E4B-it` | all 24 | card's ASR and "transcribe then translate" prompts; the translation is the text after the `<language>:` marker |
| `microsoft/Phi-4-multimodal-instruct` | ASR en de es fr it pt; translation to and from English for de es fr it pt | "Transcribe the audio clip into text." / "Translate the audio to <language>." |
| `nvidia/canary-1b-v2` | 25 European languages incl. 23 of the 24 here (no Irish); translation to and from English only | NeMo `transcribe(source_lang, target_lang)` |

**Text-only ceilings** (kind `reference`, master keys `Text-Qwen3.5-2B`, `Text-EuroLLM-1.7B`, `Text-Llama-3.2-1B`; the
Instruct variants, Qwen with thinking off, EuroLLM the 2512 release): the model translates the reference transcript of each
FLEURS test clip (`source_text` in the manifest), no audio, with the `repeatfirst` translate wording, greedy, 384 new tokens at
most. Sets `fleurs24-st-xen-test-text` and `fleurs24-st-enx-test-text`, fmt `textsrc`, metrics chrF++ and spBLEU (cased and
lowercased) per language, same scorer as the speech sets. Script: `scripts/text_translate.py` in melt-eval. Runaway outputs are
scored as they are (Llama en->X: 1,294 of 14,457 hit the cap). Published numbers of other public models (Apertus,
Kimi-Audio, Qwen2-Audio, Qwen2.5-Omni) are cited in the master for FLEURS WER only (`fleurs-reported`).

**Our checkpoints** (`kind=ours`, fmt `repeatfirst`): Whisper-large-v3 → MLP adapter → decoder (Qwen3.5-2B or
EuroLLM-1.7B-Instruct-2512), four regimes: SFT-EU24 (encoder, adapter and decoder on ASR + ST), CE24 (adapter-only,
gold transcripts), SD24 (adapter-only self-distillation) and SD-SIFT24 (the AZeroS recipe at 24 languages), plus
start models A and A′. Their FLEURS dev numbers are in the master; **their test-set runs are scheduled (2026-10-18)
and not yet in it.** One training run per cell, no seeds.

## 5. How a baseline run works (melt-eval)

Runs use melt-eval (`MELT-proj/eval`, built on `inspect_ai`). The baseline providers (Gemma 4, Phi-4-multimodal,
canary) and the per-language config are on a local, unpushed branch `baselines-gemma4-phi4-canary` in the artemis
checkout `/mnt/scratch-artemis/giuseppe/melt-proj/melt-eval-baselines` (decision 2026-10-18: keep local for now).

- Config: `configs/eval/baselines-gemma-phi-canary.yaml`. One **job** is one model on one slice; slices are
  per-language for the ASR sets (`lang:`) and per pair for translation (`dataset_id:`). A model's `languages:` list
  and `skip:` list keep it off what it does not cover.
- Run: `python scripts/evaluate.py run CONFIG --site artemis --wait`, with `MAX_QUEUED=4 MAX_QUEUED_OWN=1` on the
  `gpu-short` QOS. `run` starts only jobs without a successful log, so rerunning retries exactly what is missing.
  Outputs: `/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/baselines-gemma-phi-canary/logs/<model>/<benchmark>/<slice>/*.json`.
- Clips longer than a model's stated limit (30 s Gemma, 40 s Phi-4) are passed whole and counted in the sample
  metadata (`audio_over_model_limit_seconds`). No clip in these sets exceeded it.
- Environment traps: `inspect` splits every `-T key=value` at commas, so instructions with commas travel as files
  (`instruction_file`); Phi-4 needs the card's pins (transformers 4.48.2, peft 0.13.2) layered in
  `pyextras/pre-tf5-shim`; canary needs a NeMo venv with lhotse 2 (`pyextras/canary-shim`); Gemma 4 needs
  transformers 5 and torchvision from `pyextras`. All under `/mnt/scratch-artemis/giuseppe/melt-data/pyextras/`.
  The melt-eval venv is `/mnt/scratch-artemis/giuseppe/venvs/melteval`, run with `HF_HUB_OFFLINE=1`.

## 6. Getting numbers into the master

1. Register each new set once: `master.py add-set` (proposed names: the six test sets in §2; fmt `native` for
   baselines, `repeatfirst` for ours).
2. Read the JSON logs into cells (`<set>.<fmt>.<metric>.<lang or direction>`): WER/CER from the log's scorer metrics
   (`wer_<lang>`, `cer_<lang>`); chrF++ cased and lowercased and spBLEU for translation. Our checkpoints:
   `charts/ingest_eu24_report.py` on `eu24_report.json` (`projects/self-distill/eval/eu24_report.py --split test`).
3. Follow `charts/PROTOCOL.md` §4: dry run must say "0 changed" on rows you did not touch, validate, commit the three
   data files, push to `self-distill-whisper`.
4. Changes to schema, metric list or protocol are the PI's call: post them on `board.md`.

## 7. Where things stand (2026-10-19)

- Baselines: all 196 jobs on the six test sets finished and are in the master (Gemma 4, Phi-4, canary).
- Our checkpoints: all 8 final checkpoints on FLEURS-24 dev and on the six test sets are in the master.
- Text-only ceilings (Qwen3.5-2B, EuroLLM-1.7B, Llama-3.2-1B) are in the master, both directions.
- Not done: start models A and A' on the test sets, COMET, Google Drive copy of the master (deferred by the PI).
- Baseline results were not checked against the numbers on each model card (deferred by the PI).

## 8. Open items for the next session

Start models A and A' on the test sets if the paper table needs them; add `SFT-EU24-EuroLLM`/`CE24-EuroLLM` keys to
`eval/eu24_report.py` and `eval/eu24_jobs.py`; update `charts/PROTOCOL.md` §8 as items land (propose, do not edit
schema); the melt-eval branch is proposed as a PR, merge is the PI's call.
