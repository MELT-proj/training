# EU24 comparison on FLEURS dev: resource tiers, chrF++, spBLEU, selection metric

Built by `eval/eu24_report.py` and `eval/eu24_slide_tables.py` from the melt-eval logs of the step-6 runs (see `../../README.md`).
Per-language tables for every column: `eu24-dev-report.md`.

- **Set.** FLEURS-24 dev, 100 clips per language or direction (ASR 2,400 clips / 24 languages; X->en 2,300 / 23 sources; en->X 2,300 / 23 targets).
  The full test sets were not run. FLEURS train is part of the SFT-EU24 and SD mixes (most of the tail languages' ASR data), so for the mid and low tiers
  dev is closer to in-domain than for the high tier.
- **Prompts.** The repeat-first instructions the SD runs were trained on, the same for every model (SFT-EU24 scores the same under its own random
  templates); Table 5 swaps them for the random task templates. Final checkpoints, greedy decoding, 256-token cap.
- **ASR.** WER and CER with the basic normaliser, corpus-level per language, then the plain mean over the languages of a group. Values above 100% are real.
- **Translation.** sacrebleu 2.5.1 through melt-eval PR #28: chrF++ = character 6-grams + word 1- and 2-grams (`nc:6|nw:2`); spBLEU = BLEU over the
  flores200 SentencePiece model. Per direction, then the plain mean. **Lower-cased scoring is the primary one**: SFT-EU24, CE24 and A write 97-100% of their
  translations without a capital letter (Table 4, last column), SD24 and SD-SIFT24 inherit the decoder's casing, and cased scoring would measure that habit
  instead of the translation. Table 4 gives the cased scores.
- **Tiers** by hours of transcribed audio per language in the MELT pools (the per-entry hours in `../../sft/SFT-EU24.yaml`): high >= 1,000 h: en es fr de it nl pt
  (6 for translation: en is neither source nor target); mid 100-1,000 h: pl hu fi cs ro sv; low < 100 h: sk bg lv da lt el et hr sl ga mt.
  "5 trained" = en de es fr it.
- **Selection metric** (Table 6): `projects/ablation-campaign/selection-metric/score.py`, applied to its own frozen sets re-built on artemis from the
  de-duplicated shar tree (`selection-id-{en,de,es,fr,it}`, `selection-fleurs26-dev200`; specs on melt-eval branch `selection-metric-sets`).
- **Models.** Qwen3.5-2B decoder: A (aligned start), SFT-EU24 (full fine-tune, reference), CE24 (adapter, gold CE), SD24 (adapter, instruction-mixed SD), SD-SIFT24 (adapter, AZeroS). EuroLLM-1.7B-Instruct-2512 decoder: A' (start), SD24, SD-SIFT24.

**Table 1. ASR on FLEURS dev, % (macro-average over the languages of each group; lower is better)**

| decoder | model | WER all 24 | WER high (7) | WER mid (6) | WER low (11) | WER 5 trained languages | CER all 24 | CER high | CER mid | CER low |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | A (start) | 97.0 | 29.0 | 115.1 | 130.5 | 8.2 | 55.0 | 12.4 | 63.0 | 77.8 |
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | 15.5 | 7.2 | 11.5 | 23.0 | 6.5 | 6.5 | 3.2 | 4.2 | 9.9 |
| Qwen3.5-2B | CE24 (adapter, gold CE) | 19.7 | 7.7 | 15.7 | 29.6 | 6.8 | 8.3 | 3.4 | 5.4 | 12.9 |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | 20.9 | 7.7 | 15.7 | 32.2 | 6.7 | 9.0 | 3.4 | 5.4 | 14.5 |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | 28.9 | 10.5 | 22.7 | 44.1 | 9.5 | 13.8 | 4.0 | 9.4 | 22.5 |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | 103.4 | 29.6 | 132.1 | 134.8 | 8.9 | 63.2 | 13.0 | 80.7 | 85.5 |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | 26.3 | 8.2 | 17.7 | 42.5 | 7.5 | 14.4 | 3.9 | 7.6 | 24.8 |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | 73.6 | 36.4 | 53.7 | 108.2 | 39.6 | 54.1 | 29.9 | 38.2 | 78.2 |

**Table 2. chrF++ on FLEURS dev, lowercased (macro-average over directions; higher is better)**

| decoder | model | X→en all 23 | high (6) | mid (6) | low (11) | en→X all 23 | high (6) | mid (6) | low (11) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | A (start) | 19.3 | 21.3 | 19.6 | 18.1 | 17.7 | 25.3 | 16.8 | 14.0 |
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | 52.6 | 57.8 | 55.5 | 48.2 | 40.2 | 50.6 | 41.8 | 33.7 |
| Qwen3.5-2B | CE24 (adapter, gold CE) | 16.3 | 21.2 | 16.7 | 13.4 | 15.6 | 19.7 | 15.9 | 13.3 |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | 48.3 | 56.2 | 50.6 | 42.9 | 42.0 | 50.8 | 42.3 | 37.0 |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | 49.0 | 55.8 | 51.5 | 44.0 | 41.4 | 50.9 | 42.2 | 35.8 |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | 18.8 | 21.0 | 18.8 | 17.6 | 15.4 | 19.2 | 15.7 | 13.1 |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | 46.7 | 53.2 | 49.9 | 41.3 | 44.3 | 51.1 | 43.5 | 41.0 |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | 45.5 | 52.9 | 48.6 | 39.8 | 42.1 | 50.0 | 41.3 | 38.3 |

**Table 3. spBLEU on FLEURS dev, lowercased (macro-average over directions; higher is better)**

| decoder | model | X→en all 23 | high (6) | mid (6) | low (11) | en→X all 23 | high (6) | mid (6) | low (11) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | A (start) | 1.9 | 2.0 | 2.1 | 1.7 | 3.0 | 6.6 | 2.0 | 1.6 |
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | 29.4 | 34.8 | 32.0 | 25.1 | 19.3 | 27.1 | 20.0 | 14.5 |
| Qwen3.5-2B | CE24 (adapter, gold CE) | 1.4 | 1.9 | 1.2 | 1.2 | 1.3 | 2.1 | 1.0 | 0.9 |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | 25.4 | 33.2 | 27.5 | 20.1 | 20.0 | 28.4 | 20.1 | 15.4 |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | 26.2 | 32.9 | 28.2 | 21.5 | 19.5 | 28.3 | 20.4 | 14.3 |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | 1.3 | 1.6 | 1.4 | 1.1 | 1.3 | 1.8 | 1.1 | 1.1 |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | 22.3 | 28.5 | 25.3 | 17.2 | 23.3 | 29.1 | 23.2 | 20.2 |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | 20.7 | 27.6 | 23.0 | 15.6 | 21.1 | 27.7 | 20.8 | 17.8 |

**Table 4. The same two metrics, cased (all 23 directions)**

| decoder | model | chrF++ X→en | chrF++ en→X | spBLEU X→en | spBLEU en→X | translations with no capital letter, X→en / en→X |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | A (start) | 18.0 | 16.7 | 0.8 | 2.1 | 99% / 100% |
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | 48.2 | 37.7 | 22.5 | 15.6 | 100% / 100% |
| Qwen3.5-2B | CE24 (adapter, gold CE) | 15.1 | 14.7 | 0.2 | 0.5 | 100% / 100% |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | 47.4 | 41.3 | 23.6 | 19.3 | 0% / 1% |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | 48.0 | 40.8 | 24.3 | 18.8 | 0% / 3% |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | 17.6 | 14.5 | 0.4 | 0.4 | 97% / 99% |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | 45.6 | 43.5 | 20.4 | 22.3 | 2% / 6% |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | 44.3 | 41.2 | 18.9 | 20.1 | 7% / 17% |

**Table 5. Same models under the random task templates instead of the repeat-first prompts (lowercased; repeat-first → random)**

| decoder | model | WER all 24 (%) | chrF++ X→en | chrF++ en→X | spBLEU X→en | spBLEU en→X |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | 15.5 → 15.5 | 52.6 → 52.3 | 40.2 → 40.3 | 29.4 → 29.4 | 19.3 → 19.2 |
| Qwen3.5-2B | CE24 (adapter, gold CE) | 19.7 → 20.5 | 16.3 → 16.3 | 15.6 → 19.9 | 1.4 → 1.4 | 1.3 → 4.7 |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | 20.9 → 24.0 | 48.3 → 38.2 | 42.0 → 41.0 | 25.4 → 17.0 | 20.0 → 19.4 |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | 28.9 → 250.0 | 49.0 → 48.8 | 41.4 → 41.3 | 26.2 → 25.2 | 19.5 → 19.2 |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | 103.4 → 96.8 | 18.8 → 20.3 | 15.4 → 15.5 | 1.3 → 2.2 | 1.3 → 1.4 |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | 26.3 → 65.2 | 46.7 → 34.7 | 44.3 → 35.3 | 22.3 → 13.0 | 23.3 → 16.8 |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | 73.6 → 305.8 | 45.5 → 35.0 | 42.1 → 32.8 | 20.7 → 10.1 | 21.1 → 10.2 |

**Table 6. The campaign's selection metric (single ranking score; lower is better)**

| decoder | model | score | ID (5 trained languages, held-out CV22/MLS/VoxPopuli) | OOD-train (same 5, FLEURS dev) | OOD-related (pt ro nl da sv) | OOD-latin* | OOD-script* | runaway clips: in-domain / FLEURS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | A (start) | **0.118** | 0.054 | 0.040 | 0.421 | 0.670 | 1.000 | 6/3000 / 406/5171 |
| Qwen3.5-2B | SFT-EU24 (full fine-tune) | **0.038** | 0.053 | 0.026 | 0.043 | 0.043 | 0.528 | 6/3000 / 38/5171 |
| Qwen3.5-2B | CE24 (adapter, gold CE) | **0.042** | 0.056 | 0.031 | 0.050 | 0.058 | 0.485 | 8/3000 / 36/5171 |
| Qwen3.5-2B | SD24 (adapter, self-distilled) | **0.043** | 0.059 | 0.031 | 0.049 | 0.062 | 0.498 | 8/3000 / 39/5171 |
| Qwen3.5-2B | SD-SIFT24 (adapter, AZeroS) | **0.056** | 0.072 | 0.037 | 0.075 | 0.118 | 0.481 | 8/3000 / 72/5171 |
| EuroLLM-1.7B-Instruct-2512 | A' (start) | **0.149** | 0.066 | 0.052 | 0.531 | 0.809 | 1.000 | 9/3000 / 544/5171 |
| EuroLLM-1.7B-Instruct-2512 | SD24 (adapter, self-distilled) | **0.053** | 0.069 | 0.039 | 0.059 | 0.109 | 0.447 | 9/3000 / 85/5171 |
| EuroLLM-1.7B-Instruct-2512 | SD-SIFT24 (adapter, AZeroS) | **0.301** | 0.474 | 0.210 | 0.239 | 0.490 | 0.771 | 388/3000 / 499/5171 |

score = (1.0 ID + 1.5 OOD-train + 0.6 OOD-related) / 3.1; each group is the median over its languages of corpus CER clipped at 1. *reported only, not in the score (OOD-latin: the other 12 Latin-script EU languages; OOD-script: bg el ru uk, of which ru and uk were never trained).

