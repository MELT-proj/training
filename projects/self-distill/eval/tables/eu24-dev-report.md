# EU24 results on FLEURS dev

Generated 2026-10-15 18:54 UTC. Prompt format `fmt-repeatfirst`. Eval code: melt-eval PR #28 @ `cf6f9e8`. Sets: fleurs24-asr-dev, fleurs24-st-xen-dev, fleurs24-st-enx-dev. FLEURS-24 dev, 100 clips per language or direction (ASR 2,400; X→en 2,300; en→X 2,300).

## Language tiers

Rule: high: >= 1,000 h of transcribed audio in the MELT pools; mid: 100-1,000 h; low: < 100 h. Hours are the transcribed audio held per language in the MELT pools (CV22, MLS, VoxPopuli, Granary, FLEURS train), the number SFT-EU24's data mix was built from.

- **high** (7): en (144,001 h), es (28,200 h), fr (14,371 h), de (11,128 h), it (6,734 h), nl (3,028 h), pt (2,101 h)
- **mid** (6): pl (931 h), hu (274 h), fi (194 h), cs (160 h), ro (131 h), sv (110 h)
- **low** (11): sk (54 h), bg (35 h), lv (30 h), da (29 h), lt (28 h), el (27 h), et (24 h), hr (21 h), sl (15 h), ga (13 h), mt (12 h)

## 1. WER on FLEURS dev, per language (%)

| lang (tier) | A | SFT-EU24 | CE24 | SD24 | SD-SIFT24 | A-EuroLLM | SD-SIFT24-EuroLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| en (high) | 8.0 | 7.5 | 6.7 | 6.8 | 7.8 | 8.2 | 64.6 |
| es (high) | 7.4 | 6.7 | 6.0 | 6.2 | 7.7 | 7.4 | 16.0 |
| fr (high) | 9.4 | 6.9 | 7.8 | 7.6 | 10.7 | 11.5 | 18.6 |
| de (high) | 9.5 | 6.8 | 8.1 | 7.9 | 15.6 | 11.3 | 69.2 |
| it (high) | 6.7 | 4.4 | 5.3 | 5.2 | 5.7 | 6.0 | 29.7 |
| nl (high) | 96.4 | 10.5 | 12.6 | 13.2 | 16.4 | 91.3 | 32.8 |
| pt (high) | 65.5 | 7.5 | 7.4 | 7.2 | 9.2 | 71.6 | 24.1 |
| pl (mid) | 102.5 | 7.7 | 11.6 | 10.6 | 16.5 | 111.4 | 57.7 |
| hu (mid) | 192.7 | 12.6 | 17.7 | 18.4 | 24.3 | 172.3 | 76.8 |
| fi (mid) | 121.1 | 14.4 | 17.7 | 19.6 | 31.0 | 166.2 | 68.9 |
| cs (mid) | 95.0 | 12.9 | 18.1 | 17.1 | 22.7 | 124.4 | 45.6 |
| ro (mid) | 94.1 | 11.9 | 14.5 | 14.0 | 18.6 | 105.5 | 42.2 |
| sv (mid) | 85.0 | 9.5 | 14.4 | 14.7 | 22.9 | 112.6 | 30.9 |
| sk (low) | 100.4 | 10.7 | 15.1 | 15.6 | 24.4 | 122.3 | 47.0 |
| bg (low) | 148.3 | 27.3 | 16.4 | 16.8 | 29.3 | 118.0 | 48.9 |
| lv (low) | 111.0 | 14.8 | 26.0 | 27.3 | 36.2 | 136.8 | 58.8 |
| da (low) | 95.8 | 20.0 | 19.9 | 20.5 | 26.2 | 96.9 | 32.4 |
| lt (low) | 133.3 | 22.4 | 33.8 | 33.8 | 38.5 | 157.1 | 79.2 |
| el (low) | 143.5 | 21.9 | 19.8 | 21.2 | 28.8 | 123.7 | 71.1 |
| et (low) | 83.8 | 15.7 | 26.0 | 26.6 | 31.4 | 143.7 | 93.3 |
| hr (low) | 104.1 | 14.1 | 15.2 | 16.8 | 27.3 | 126.9 | 87.2 |
| sl (low) | 161.8 | 16.7 | 24.1 | 25.6 | 32.4 | 139.0 | 71.4 |
| ga (low) | 197.6 | 57.7 | 77.9 | 84.4 | 115.5 | 190.5 | 403.3 |
| mt (low) | 155.4 | 31.7 | 51.8 | 65.3 | 95.4 | 127.7 | 197.8 |

## 2. Selection metric (the campaign's single ranking score, lower is better)

Weighted median of per-language CER clipped at 1: (1.0 ID + 1.5 OOD-train + 0.6 OOD-related) / 3.1. ID = the five trained languages on held-out CV22 / MLS / VoxPopuli (600 utterances each); the OOD groups are FLEURS dev, 200 utterances per language (ru and uk included in the reported-only groups; the models were not trained on them).

| model | score | ID | OOD-train | OOD-related | OOD-latin* | OOD-script* | runaways ID / FLEURS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | **0.118** | 0.054 | 0.040 | 0.421 | 0.670 | 1.000 | 6/3000 / 406/5171 |
| SFT-EU24 | **0.038** | 0.053 | 0.026 | 0.043 | 0.043 | 0.528 | 6/3000 / 38/5171 |
| CE24 | **0.042** | 0.056 | 0.031 | 0.050 | 0.058 | 0.485 | 8/3000 / 36/5171 |
| SD24 | **0.043** | 0.059 | 0.031 | 0.049 | 0.062 | 0.498 | 8/3000 / 39/5171 |
| SD-SIFT24 | **0.056** | 0.072 | 0.037 | 0.075 | 0.118 | 0.481 | 8/3000 / 72/5171 |
| A-EuroLLM | **0.149** | 0.066 | 0.052 | 0.531 | 0.809 | 1.000 | 9/3000 / 544/5171 |
| SD-SIFT24-EuroLLM | **0.301** | 0.474 | 0.210 | 0.239 | 0.490 | 0.771 | 388/3000 / 499/5171 |

*reported only, not in the score.

## 3. WER and CER on FLEURS dev by resource tier (%, macro-average over languages)

| model | WER all-24 | high | mid | low | trained-5 | CER all-24 | high | mid | low | trained-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 97.0 | 29.0 | 115.1 | 130.5 | 8.2 | 55.0 | 12.4 | 63.0 | 77.8 | 4.1 |
| SFT-EU24 | 15.5 | 7.2 | 11.5 | 23.0 | 6.5 | 6.5 | 3.2 | 4.2 | 9.9 | 2.9 |
| CE24 | 19.7 | 7.7 | 15.7 | 29.6 | 6.8 | 8.3 | 3.4 | 5.4 | 12.9 | 3.1 |
| SD24 | 20.9 | 7.7 | 15.7 | 32.2 | 6.7 | 9.0 | 3.4 | 5.4 | 14.5 | 3.1 |
| SD-SIFT24 | 28.9 | 10.5 | 22.7 | 44.1 | 9.5 | 13.8 | 4.0 | 9.4 | 22.5 | 3.8 |
| A-EuroLLM | 103.4 | 29.6 | 132.1 | 134.8 | 8.9 | 63.2 | 13.0 | 80.7 | 85.5 | 4.6 |
| SD-SIFT24-EuroLLM | 73.6 | 36.4 | 53.7 | 108.2 | 39.6 | 54.1 | 29.9 | 38.2 | 78.2 | 32.8 |

## 4. Translation: chrF++ and spBLEU on FLEURS dev by resource tier (macro-average over directions)

chrF++ = character 6-grams plus word 1- and 2-grams (sacrebleu `nc:6|nw:2`); spBLEU = BLEU over the flores200 SentencePiece model. **Lowercased scoring is the primary one**: the standard-training models (A, SFT-EU24, CE24) write their translations without any capital letter (share below), a habit inherited from their training targets, so cased scoring would measure that habit instead of the translation. SD models inherit the decoder's natural casing. The cased scores follow for reference.

| share of translations with no capital letter | X→en | en→X |
|---|---:|---:|
| A | 99% | 100% |
| SFT-EU24 | 100% | 100% |
| CE24 | 100% | 100% |
| SD24 | 0% | 1% |
| SD-SIFT24 | 0% | 3% |
| A-EuroLLM | 97% | 99% |
| SD-SIFT24-EuroLLM | 7% | 17% |

### 4.1 chrF++, lowercased (primary)

| model | X→en all-23 | high | mid | low | en→X all-23 | high | mid | low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 19.3 | 21.3 | 19.6 | 18.1 | 17.7 | 25.3 | 16.8 | 14.0 |
| SFT-EU24 | 52.6 | 57.8 | 55.5 | 48.2 | 40.2 | 50.6 | 41.8 | 33.7 |
| CE24 | 16.3 | 21.2 | 16.7 | 13.4 | 15.6 | 19.7 | 15.9 | 13.3 |
| SD24 | 48.3 | 56.2 | 50.6 | 42.9 | 42.0 | 50.8 | 42.3 | 37.0 |
| SD-SIFT24 | 49.0 | 55.8 | 51.5 | 44.0 | 41.4 | 50.9 | 42.2 | 35.8 |
| A-EuroLLM | 18.8 | 21.0 | 18.8 | 17.6 | 15.4 | 19.2 | 15.7 | 13.1 |
| SD-SIFT24-EuroLLM | 45.5 | 52.9 | 48.6 | 39.8 | 42.1 | 50.0 | 41.3 | 38.3 |

### 4.2 spBLEU, lowercased (primary)

| model | X→en all-23 | high | mid | low | en→X all-23 | high | mid | low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.9 | 2.0 | 2.1 | 1.7 | 3.0 | 6.6 | 2.0 | 1.6 |
| SFT-EU24 | 29.4 | 34.8 | 32.0 | 25.1 | 19.3 | 27.1 | 20.0 | 14.5 |
| CE24 | 1.4 | 1.9 | 1.2 | 1.2 | 1.3 | 2.1 | 1.0 | 0.9 |
| SD24 | 25.4 | 33.2 | 27.5 | 20.1 | 20.0 | 28.4 | 20.1 | 15.4 |
| SD-SIFT24 | 26.2 | 32.9 | 28.2 | 21.5 | 19.5 | 28.3 | 20.4 | 14.3 |
| A-EuroLLM | 1.3 | 1.6 | 1.4 | 1.1 | 1.3 | 1.8 | 1.1 | 1.1 |
| SD-SIFT24-EuroLLM | 20.7 | 27.6 | 23.0 | 15.6 | 21.1 | 27.7 | 20.8 | 17.8 |

### 4.3 chrF++, cased

| model | X→en all-23 | high | mid | low | en→X all-23 | high | mid | low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18.0 | 19.7 | 18.2 | 16.9 | 16.7 | 23.7 | 16.0 | 13.2 |
| SFT-EU24 | 48.2 | 52.8 | 50.8 | 44.3 | 37.7 | 46.4 | 39.5 | 31.9 |
| CE24 | 15.1 | 19.6 | 15.6 | 12.5 | 14.7 | 18.4 | 15.0 | 12.6 |
| SD24 | 47.4 | 55.2 | 49.7 | 41.9 | 41.3 | 49.9 | 41.6 | 36.5 |
| SD-SIFT24 | 48.0 | 54.7 | 50.6 | 43.0 | 40.8 | 50.0 | 41.6 | 35.3 |
| A-EuroLLM | 17.6 | 19.6 | 17.6 | 16.5 | 14.5 | 18.0 | 14.9 | 12.4 |
| SD-SIFT24-EuroLLM | 44.3 | 51.6 | 47.3 | 38.8 | 41.2 | 48.9 | 40.4 | 37.4 |

### 4.4 spBLEU, cased

| model | X→en all-23 | high | mid | low | en→X all-23 | high | mid | low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.8 | 0.7 | 1.1 | 0.7 | 2.1 | 5.0 | 1.3 | 0.9 |
| SFT-EU24 | 22.5 | 26.9 | 24.5 | 19.0 | 15.6 | 20.7 | 17.1 | 12.1 |
| CE24 | 0.2 | 0.4 | 0.2 | 0.2 | 0.5 | 0.9 | 0.3 | 0.3 |
| SD24 | 23.6 | 31.3 | 25.5 | 18.4 | 19.3 | 27.0 | 19.6 | 14.9 |
| SD-SIFT24 | 24.3 | 30.7 | 26.1 | 19.7 | 18.8 | 27.0 | 19.6 | 13.8 |
| A-EuroLLM | 0.4 | 0.5 | 0.5 | 0.4 | 0.4 | 0.7 | 0.4 | 0.3 |
| SD-SIFT24-EuroLLM | 18.9 | 25.5 | 20.7 | 14.3 | 20.1 | 26.1 | 19.9 | 16.9 |

## 5. Translation per language (lowercased)

### chrF++, X→en

| lang (tier) | A | SFT-EU24 | CE24 | SD24 | SD-SIFT24 | A-EuroLLM | SD-SIFT24-EuroLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| es (high) | 20.0 | 53.6 | 20.3 | 54.2 | 53.0 | 20.0 | 48.2 |
| fr (high) | 21.6 | 58.0 | 22.0 | 58.2 | 57.6 | 21.4 | 53.3 |
| de (high) | 21.1 | 61.8 | 21.0 | 58.6 | 58.4 | 20.7 | 54.5 |
| it (high) | 21.3 | 57.9 | 21.5 | 56.6 | 55.4 | 21.3 | 52.5 |
| nl (high) | 20.9 | 52.6 | 20.7 | 49.4 | 49.6 | 20.4 | 49.5 |
| pt (high) | 22.7 | 62.8 | 21.3 | 60.1 | 60.9 | 22.2 | 59.8 |
| pl (mid) | 19.4 | 51.6 | 14.5 | 48.0 | 48.3 | 18.5 | 46.4 |
| hu (mid) | 15.1 | 50.5 | 14.1 | 42.7 | 43.6 | 16.8 | 42.1 |
| fi (mid) | 16.6 | 52.9 | 15.2 | 44.6 | 46.8 | 15.2 | 44.4 |
| cs (mid) | 19.8 | 55.8 | 15.2 | 53.3 | 53.8 | 18.5 | 52.0 |
| ro (mid) | 22.2 | 59.3 | 21.1 | 56.5 | 57.1 | 22.2 | 50.5 |
| sv (mid) | 24.7 | 62.7 | 20.3 | 58.2 | 59.1 | 21.8 | 56.1 |
| sk (low) | 18.5 | 55.6 | 15.3 | 53.2 | 54.2 | 19.4 | 51.9 |
| bg (low) | 20.0 | 56.6 | 2.2 | 52.3 | 54.5 | 20.6 | 51.0 |
| lv (low) | 15.8 | 47.9 | 14.2 | 40.3 | 40.3 | 15.1 | 35.8 |
| da (low) | 23.5 | 60.8 | 20.8 | 55.7 | 57.0 | 20.8 | 55.1 |
| lt (low) | 16.7 | 43.9 | 15.1 | 38.7 | 41.3 | 15.5 | 34.6 |
| el (low) | 17.1 | 48.6 | 2.2 | 47.9 | 48.1 | 17.2 | 45.4 |
| et (low) | 18.4 | 51.6 | 16.1 | 40.0 | 40.2 | 15.6 | 36.2 |
| hr (low) | 19.1 | 53.3 | 15.8 | 50.6 | 52.7 | 18.5 | 47.7 |
| sl (low) | 17.5 | 49.1 | 15.3 | 42.5 | 43.4 | 17.9 | 38.3 |
| ga (low) | 13.7 | 19.0 | 13.0 | 16.6 | 15.9 | 13.5 | 11.3 |
| mt (low) | 18.4 | 44.3 | 17.4 | 33.6 | 36.0 | 19.4 | 30.4 |

### spBLEU, X→en

| lang (tier) | A | SFT-EU24 | CE24 | SD24 | SD-SIFT24 | A-EuroLLM | SD-SIFT24-EuroLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| es (high) | 0.9 | 29.0 | 1.3 | 30.0 | 28.9 | 1.3 | 22.5 |
| fr (high) | 1.6 | 33.1 | 1.7 | 34.6 | 33.7 | 1.3 | 25.1 |
| de (high) | 2.2 | 41.0 | 2.3 | 37.5 | 36.9 | 1.5 | 28.4 |
| it (high) | 1.4 | 33.1 | 1.5 | 32.8 | 31.1 | 1.5 | 26.4 |
| nl (high) | 2.3 | 30.4 | 2.4 | 26.7 | 26.7 | 1.4 | 25.7 |
| pt (high) | 3.4 | 42.0 | 2.1 | 37.7 | 40.0 | 2.8 | 37.5 |
| pl (mid) | 1.8 | 26.5 | 0.8 | 23.6 | 23.8 | 1.0 | 19.9 |
| hu (mid) | 0.7 | 26.8 | 0.5 | 20.1 | 21.2 | 1.0 | 17.0 |
| fi (mid) | 1.2 | 29.6 | 0.9 | 22.1 | 23.2 | 0.9 | 20.2 |
| cs (mid) | 2.1 | 32.7 | 1.1 | 30.2 | 30.4 | 1.2 | 25.4 |
| ro (mid) | 1.6 | 35.7 | 1.9 | 32.8 | 33.3 | 1.7 | 24.2 |
| sv (mid) | 5.5 | 40.7 | 1.9 | 36.1 | 37.4 | 2.6 | 31.2 |
| sk (low) | 2.5 | 32.5 | 1.5 | 29.9 | 30.9 | 1.8 | 26.0 |
| bg (low) | 2.1 | 31.7 | 0.7 | 26.9 | 30.7 | 1.8 | 22.8 |
| lv (low) | 0.7 | 24.5 | 0.3 | 17.7 | 17.8 | 0.3 | 12.8 |
| da (low) | 4.2 | 38.5 | 2.8 | 30.9 | 33.4 | 2.6 | 29.7 |
| lt (low) | 1.2 | 19.5 | 0.9 | 14.8 | 18.3 | 0.8 | 10.5 |
| el (low) | 0.7 | 24.0 | 0.1 | 23.2 | 22.8 | 0.7 | 18.0 |
| et (low) | 2.1 | 27.4 | 1.6 | 18.9 | 18.1 | 0.7 | 11.2 |
| hr (low) | 2.2 | 29.7 | 1.3 | 27.8 | 29.6 | 1.1 | 21.5 |
| sl (low) | 0.7 | 26.2 | 1.1 | 19.1 | 19.9 | 0.6 | 12.2 |
| ga (low) | 0.6 | 2.5 | 0.7 | 1.3 | 1.5 | 0.6 | 0.3 |
| mt (low) | 1.3 | 19.4 | 2.0 | 10.3 | 13.0 | 1.2 | 6.7 |

### chrF++, en→X

| lang (tier) | A | SFT-EU24 | CE24 | SD24 | SD-SIFT24 | A-EuroLLM | SD-SIFT24-EuroLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| es (high) | 28.2 | 48.5 | 19.2 | 48.0 | 48.2 | 17.8 | 48.2 |
| fr (high) | 28.0 | 55.3 | 20.7 | 56.6 | 57.3 | 21.4 | 55.8 |
| de (high) | 24.4 | 55.6 | 18.3 | 51.1 | 50.8 | 18.0 | 49.7 |
| it (high) | 23.6 | 46.0 | 19.8 | 48.6 | 48.2 | 19.4 | 46.9 |
| nl (high) | 24.3 | 46.5 | 20.0 | 44.9 | 44.6 | 19.0 | 43.9 |
| pt (high) | 23.6 | 52.0 | 20.0 | 55.6 | 56.2 | 19.5 | 55.8 |
| pl (mid) | 14.1 | 39.1 | 14.0 | 38.1 | 38.1 | 13.9 | 33.2 |
| hu (mid) | 14.7 | 29.9 | 13.1 | 37.6 | 38.6 | 12.9 | 39.6 |
| fi (mid) | 14.6 | 31.1 | 14.1 | 35.2 | 34.8 | 14.0 | 35.9 |
| cs (mid) | 15.7 | 40.2 | 14.6 | 43.4 | 42.7 | 14.4 | 40.8 |
| ro (mid) | 20.7 | 51.4 | 18.8 | 48.9 | 49.1 | 18.4 | 47.6 |
| sv (mid) | 21.2 | 58.9 | 20.7 | 50.4 | 50.3 | 20.6 | 50.6 |
| sk (low) | 15.1 | 39.2 | 15.0 | 42.8 | 41.8 | 14.8 | 41.7 |
| bg (low) | 1.4 | 20.0 | 1.7 | 47.5 | 47.1 | 1.5 | 42.6 |
| lv (low) | 13.7 | 48.8 | 13.8 | 32.1 | 31.5 | 13.7 | 37.6 |
| da (low) | 25.8 | 46.9 | 21.7 | 51.3 | 50.6 | 21.4 | 53.9 |
| lt (low) | 15.0 | 28.7 | 14.4 | 36.1 | 34.5 | 14.2 | 34.8 |
| el (low) | 2.1 | 16.6 | 1.6 | 37.2 | 37.1 | 1.5 | 37.1 |
| et (low) | 16.3 | 50.3 | 16.5 | 32.2 | 32.1 | 16.2 | 33.0 |
| hr (low) | 19.0 | 38.4 | 16.3 | 44.0 | 42.9 | 15.8 | 34.8 |
| sl (low) | 15.7 | 51.6 | 15.4 | 39.5 | 39.4 | 15.2 | 38.1 |
| ga (low) | 13.9 | 13.2 | 14.0 | 20.4 | 16.9 | 14.0 | 32.9 |
| mt (low) | 15.7 | 17.1 | 15.9 | 24.1 | 20.0 | 16.1 | 34.2 |

### spBLEU, en→X

| lang (tier) | A | SFT-EU24 | CE24 | SD24 | SD-SIFT24 | A-EuroLLM | SD-SIFT24-EuroLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| es (high) | 7.8 | 24.2 | 2.0 | 24.4 | 24.6 | 0.8 | 25.4 |
| fr (high) | 8.6 | 32.2 | 2.0 | 36.3 | 36.4 | 3.1 | 36.0 |
| de (high) | 5.7 | 34.3 | 2.1 | 28.3 | 27.6 | 1.6 | 25.5 |
| it (high) | 5.7 | 21.5 | 2.0 | 25.8 | 25.8 | 1.7 | 24.1 |
| nl (high) | 5.7 | 21.9 | 2.7 | 21.6 | 20.6 | 2.2 | 20.1 |
| pt (high) | 6.0 | 28.8 | 2.0 | 33.8 | 35.0 | 1.7 | 35.0 |
| pl (mid) | 0.7 | 18.6 | 0.6 | 17.6 | 17.7 | 0.6 | 14.9 |
| hu (mid) | 1.7 | 7.7 | 0.3 | 14.8 | 15.5 | 0.5 | 17.1 |
| fi (mid) | 0.5 | 7.0 | 0.4 | 10.4 | 9.9 | 0.3 | 12.2 |
| cs (mid) | 2.2 | 18.8 | 0.8 | 23.6 | 23.8 | 0.8 | 24.0 |
| ro (mid) | 3.8 | 29.1 | 1.2 | 27.5 | 28.2 | 0.8 | 28.5 |
| sv (mid) | 3.2 | 38.8 | 2.8 | 26.9 | 27.1 | 3.4 | 28.1 |
| sk (low) | 1.2 | 16.9 | 1.0 | 20.8 | 19.1 | 1.0 | 22.0 |
| bg (low) | 0.1 | 6.0 | 0.1 | 26.8 | 25.8 | 0.1 | 25.3 |
| lv (low) | 0.5 | 28.2 | 0.3 | 9.5 | 9.5 | 0.2 | 17.0 |
| da (low) | 7.1 | 21.8 | 3.5 | 27.0 | 25.7 | 4.3 | 30.5 |
| lt (low) | 1.0 | 6.3 | 0.5 | 12.9 | 10.2 | 0.5 | 12.3 |
| el (low) | 0.6 | 5.1 | 0.2 | 18.6 | 17.8 | 0.0 | 20.7 |
| et (low) | 1.2 | 27.8 | 1.0 | 8.6 | 8.1 | 1.6 | 11.8 |
| hr (low) | 3.3 | 14.4 | 1.6 | 21.4 | 20.4 | 1.4 | 12.5 |
| sl (low) | 1.0 | 30.2 | 0.5 | 16.6 | 16.2 | 0.3 | 17.3 |
| ga (low) | 0.7 | 0.6 | 0.7 | 2.0 | 1.0 | 0.8 | 10.0 |
| mt (low) | 0.9 | 2.7 | 0.9 | 5.6 | 3.6 | 1.4 | 15.9 |

## Provenance

- **A** (`GOLDW-final-A`): asr: `2026-10-14T13-58-34-00-00_speech-asr_gADNpkVniWSMyeLPvFkPDC.eval`, xen: `2026-10-14T13-58-15-00-00_speech-st_empgU5UM7nVqE2Eh8hDiNZ.eval`, enx: `2026-10-14T13-58-34-00-00_speech-st_cGFfScmByex6br7iL6LYxt.eval`
- **SFT-EU24** (`SFT-EU24-from-goldw-bd60-ga4-8g-s1337`): asr: `2026-10-14T14-36-08-00-00_speech-asr_AroZa2BgLTPKnfgyixkerS.eval`, xen: `2026-10-14T14-36-06-00-00_speech-st_XaFT6RdAgJXDwb9LDHaZQY.eval`, enx: `2026-10-14T14-36-08-00-00_speech-st_ECV5xNqEFdfXLbaJB8ZZSv.eval`
- **CE24** (`SD24-ce-from-GOLDW-final-2700h-bd64-8g-s1337`): asr: `2026-10-14T20-38-36-00-00_speech-asr_AFGXMj9P3wWJr9SnNnQqgy.eval`, xen: `2026-10-14T20-38-18-00-00_speech-st_CJJweHspYRpuT8hadiis4S.eval`, enx: `2026-10-14T20-38-18-00-00_speech-st_nSUvCaX5Y2KL2bKs2tL6iZ.eval`
- **SD24** (`SD24-azeros-from-GOLDW-final-mix025-2700h-bd64-8g-s1337`): asr: `2026-10-15T13-24-11-00-00_speech-asr_j5PJZgPZ7iH2XcHGbNJoxy.eval`, xen: `2026-10-15T13-23-51-00-00_speech-st_nPYQyvsyZU39Pf39tK8qPQ.eval`, enx: `2026-10-15T13-23-51-00-00_speech-st_jkvgMXhat6Wb3ecCB4m6Ej.eval`
- **SD-SIFT24** (`SD-SIFT24-from-GOLDW-final-2700h-bd64-8g-s1337`): asr: `2026-10-15T18-21-27-00-00_speech-asr_mDRE4pFEsgMXs9XH5Yot4L.eval`, xen: `2026-10-15T18-21-27-00-00_speech-st_LtieigMxfSTGPBESC4FG4R.eval`, enx: `2026-10-15T18-21-27-00-00_speech-st_M7owWXTY6kN4icEkTB89gU.eval`
- **A-EuroLLM** (`EuroLLM-A`): asr: `2026-10-14T13-58-36-00-00_speech-asr_5cfU5E3nvzd5Tmnfxj7iJV.eval`, xen: `2026-10-14T13-58-36-00-00_speech-st_kDF6ty5yaUHhXdjroRa8zY.eval`, enx: `2026-10-14T13-58-36-00-00_speech-st_XfVLwaDBzLr2jTXbJEXabs.eval`
- **SD-SIFT24-EuroLLM** (`SD-SIFT24-from-EuroLLM-A-2700h-bd64-8g-s1337`): asr: `2026-10-15T14-27-12-00-00_speech-asr_YL28cVhGbzBymaEUWANkpu.eval`, xen: `2026-10-15T14-27-12-00-00_speech-st_duYqfjzNDoC9FBCXjPsVkK.eval`, enx: `2026-10-15T14-27-12-00-00_speech-st_bkVUuNYGaGgeXAELJjj74R.eval`

sacrebleu signatures: cased-bleu: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1`; cased-spbleu: `nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.5.1`; cased-chrf: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1`; cased-chrfpp: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.5.1`; lower-bleu: `nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.5.1`; lower-spbleu: `nrefs:1|case:lc|eff:no|tok:flores200|smooth:exp|version:2.5.1`; lower-chrf: `nrefs:1|case:lc|eff:yes|nc:6|nw:0|space:no|version:2.5.1`; lower-chrfpp: `nrefs:1|case:lc|eff:yes|nc:6|nw:2|space:no|version:2.5.1`
