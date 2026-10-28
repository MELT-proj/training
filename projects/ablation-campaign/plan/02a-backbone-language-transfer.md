# 02a — Is a backbone weak in the untrained languages, or only unseen? (plan and Step B results)

Status: **Step B run, results in §5a, for the PI to review** (plan 2026-10-24, results 2026-10-26). Follows `02-backbones.md` §5.

## 1. The question

The IFT backbones (llama1b, qwen35-2b, eurollm1.7b, base and instruct) were trained on five languages
(en, de, es, fr, it). The final model must serve all 24 EU languages. The selection score has a term on five
never-trained languages (OOD-related: pt, ro, nl, da, sv; weight 0.6/3.1). That term favours decoders pretrained on
many EU languages (EuroLLM), and the headline `sel (ASR+ST)` order follows it: EuroLLM leads because of OOD-related
(0.127–0.165 against 0.157–0.408), not because of the trained languages.

Two ways to remove the confound both fail:

- **Keep OOD-related.** It measures what the decoder was pretrained on plus zero-shot adapter transfer. It cannot tell
  "weak in this language" from "never saw this language"; Qwen may only be penalised.
- **Drop it.** The five trained languages are all high-resource, so the score would reward strength in high-resource
  languages and say nothing about the other 19. VoxPopuli dev (§2) shows this: Qwen leads there.

Neither reweighting resolves it. The decisive evidence is what each backbone does once it is **given** the languages.

## 2. What is already measured (context)

VoxPopuli validation, 300 clips each for en/de/es/fr/it, each checkpoint's own training prompt
("Transcribe this audio in {lang}.", llama3 template for llama, chatml for the rest), bf16, flash attention, artemis
h200 (logs under
`/mnt/scratch-artemis/giuseppe/melt-proj/backbones-battery/logs-vpdev/<checkpoint>/`). Corpus WER %, runaway = clip
CER > 1.

| arm | en | de | es | fr | it | mean WER | mean CER | runaway |
|---|---|---|---|---|---|---|---|---|
| llama1b-base | 7.0 | 15.0 | 8.0 | 10.8 | 13.2 | 10.8 | 7.1 | 7 |
| llama1b-ins | 7.6 | 15.1 | 6.3 | 10.4 | 13.4 | 10.6 | 6.9 | 6 |
| qwen35-2b-base | 6.8 | 14.3 | 6.1 | 10.1 | 11.2 | 9.7 | 6.4 | 6 |
| qwen35-2b-ins | 6.7 | 14.3 | 6.3 | 9.2 | 11.1 | 9.5 | 6.2 | 5 |
| eurollm1.7b-base | 7.4 | 15.7 | 6.6 | 11.5 | 12.5 | 10.8 | 7.0 | 6 |
| eurollm1.7b-ins | 7.4 | 15.1 | 7.6 | 11.0 | 12.7 | 10.8 | 7.3 | 9 |

Qwen is ~1 WER point ahead; the other four are tied. Not a score input: the selection score's in-domain term already
contains 200 VoxPopuli clips per language, and the validation split may have been in the runs' in-training eval.

## 3. Step A — what the decoders know before any adaptation (done, no new compute)

Both diagnostics already exist in the §3.7 text-prior runs of the bare backbones on `fleurs24-asr-dev`
(100 transcripts per language; `/mnt/scratch-artemis/giuseppe/melt-data/text-prior/<arm>-asr-dev.json`, raw = the
tokenizer's own encoding, no audio, no template). Bits per character is tokenizer-independent; tokens per word is
the tokenizer cost. The PI first skipped Step A, then reinstated it (2026-10-24); no job was needed.

Raw bits per character (lower is better; the two members of each pair differ only in the checkpoint):

| arm | en | de | es | fr | it | **trained mean** | pt | ro | nl | da | sv | **probe mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama1b-base | 0.987 | 1.142 | 1.198 | 1.053 | 1.177 | 1.111 | 1.256 | 1.373 | 1.274 | 1.555 | 1.449 | **1.381** |
| llama1b-ins | 1.072 | 1.251 | 1.284 | 1.157 | 1.333 | 1.219 | 1.354 | 1.588 | 1.484 | 1.906 | 1.693 | **1.605** |
| qwen35-2b-base | 0.948 | 0.988 | 1.042 | 0.894 | 0.991 | 0.973 | 1.079 | 1.203 | 1.152 | 1.335 | 1.380 | **1.230** |
| qwen35-2b-ins | 0.974 | 1.027 | 1.075 | 0.926 | 1.039 | 1.008 | 1.126 | 1.262 | 1.216 | 1.404 | 1.453 | **1.292** |
| eurollm1.7b-base | 1.006 | 0.983 | 1.036 | 0.904 | 0.963 | 0.979 | 1.061 | 1.096 | 1.069 | 1.222 | 1.206 | **1.131** |
| eurollm1.7b-ins | 1.020 | 0.984 | 1.045 | 0.908 | 0.978 | 0.987 | 1.076 | 1.098 | 1.078 | 1.243 | 1.220 | **1.143** |

Tokens per word (identical within a family: same tokenizer), trained mean / probe mean: llama 1.64 / 1.98,
qwen 1.39 / 1.68, eurollm 1.43 / 1.62.

Reading:

- **On the five trained languages Qwen and EuroLLM are tied as text models** (0.97–1.01 vs 0.98–0.99 bits per
  character); Llama is clearly behind (1.11–1.22). The decoder alone does not explain EuroLLM's lead there.
- **On the five probe languages EuroLLM leads Qwen by 0.10–0.16 bits per character** (1.13–1.14 vs 1.23–1.29), and the
  gap is widest where the languages are furthest from the trained ones (da: 1.22 vs 1.34; sv: 1.21 vs 1.38; ro: 1.10
  vs 1.20). Llama is far behind (1.38–1.61).
- **The tokenizers are not the story**: tokens per word are close for Qwen and EuroLLM (1.68 vs 1.62 on the probe
  languages), and clearly worse for Llama (1.98). Qwen's disadvantage on these languages is in the model, not the
  vocabulary.
- **The text prior does not predict the OOD-related CER.** OOD-related median CER after IFT
  (`02-backbones.md` §5): eurollm ins/base 0.127/0.165, llama base 0.157, llama ins 0.292, qwen base/ins 0.310/0.408.
  llama1b-base has the worst raw prior of the six on these languages (1.38) and yet an OOD-related CER next to
  EuroLLM's; llama1b-ins has a worse prior and a CER almost twice as high. A better text prior is no guarantee of
  transfer, and a worse one no sentence: how the adapter and the IFT interact with the decoder matters. That is what
  Step B measures.
- Caveat: 100 transcripts per language, one domain (FLEURS); differences under ~0.03 bits per character are
  within what the sets resolve.

## 4. Step B — a small, equal adaptation probe

Continue each IFT checkpoint (`checkpoint-6310`, the same weights as the backbone table) for a short run on the five
OOD-related languages, and score the same FLEURS-24 CER the selection score uses.

- **Data** (`plan/data/hours_by_language.csv`, ASR train hours): pt 2,100, nl 3,028, ro 131, sv 110, da 29. Probe
  budget **29 h per language** (da's whole pool is the limit; every other language has more), plus an equal amount
  of replay from the five trained languages so the probe does not just measure forgetting. Total ≈ 290 h of audio
  per arm. Checkpoints are saved at **10 h** and **29 h** per language, which gives a two-point learning curve.
- **Arms**: all six, identical data order and seed (Step A pruned nothing: it shows a gap but does not separate decoder from adapter).
- **Cost, derived not measured**: the Qwen IFT arm cost ~306 GPU-h for 6,729.85 h (≈0.045 GPU-h per audio hour,
  `board.md` 2026-09 entry). 290 h is ≈ 13 GPU-h of training per arm, so ≈ 20 GPU-h per arm with model load and
  evaluation, ≈ 120 GPU-h for six. Measure one arm first; the per-step overhead of short runs is not the IFT average.
- **Where**: artemis h200 or h100 (internet, GPUs in seconds). The checkpoints are already on artemis
  (`/mnt/scratch-artemis/giuseppe/ckpts/ift-backbones/`); the training code is a fresh `training` clone. Needs the five
  languages' Shar from nyx scratch, readable from artemis. An open item: whether an IFT-style resume is the right
  entry point or a new fine-tuning config, to be checked before any submission.
- **Eval**: `selection-fleurs26-dev200` for pt/ro/nl/da/sv and the five trained languages, repeat-first excluded:
  the checkpoint's own prompt, as for every other number in `02-backbones.md`.

## 5. Decision rule (fixed before any run)

Proposed; the PI sets the thresholds.

- `adapted OOD-related` = median over pt/ro/nl/da/sv of clipped CER after the 29 h probe.
- Trained-language quality `ID` is re-measured after the probe; a backbone that gains OOD-related by losing more than
  the eval-noise band (0.010) on ID is reported as such.
- If the best two backbones differ by less than the eval-noise band (0.002–0.010, README) on `adapted OOD-related`,
  the choice goes to the one stronger on trained-language ASR+ST (the existing `sel (ASR)` ID term and `sel (ST)`).
  Otherwise the lower `adapted OOD-related` wins.
- Report the learning-curve slope (10 h → 29 h) separately; a steep slope for a backbone behind at 10 h is the
  "penalised, not weak" signature.

## 5a. Step B results (run 2026-10-24 to 2026-10-26, artemis)

Six arms, 218 steps each (10 h/lang at step 75, 29 h/lang at the end), FLEURS-26 dev, 200 clips per language (5,171
clips per eval), each checkpoint's own prompt, CER clipped at 1.0, medians as in `selection-metric/score.py`. One
seed per arm. llama1bBaseT needed a rerun (a missing HF token, then a wrong output path, both launch faults).
Intermediate checkpoints lacked the processor files, which were copied in from the run root before scoring.

Median clipped CER over pt/ro/nl/da/sv (`adapted OOD-related`), and over en/de/es/fr/it on the same FLEURS clips:

| arm | before (step 0) | 10 h | 29 h | slope 10→29 h | trained langs before → 29 h |
|---|---|---|---|---|---|
| qwen35-2b-base | .3089 | .0518 | **.0472** | −.0046 | .0152 → .0168 |
| qwen35-2b-ins | .4322 | .0543 | .0489 | −.0054 | .0209 → .0175 |
| llama1b-ins | .2630 | .0597 | .0507 | −.0091 | .0195 → .0187 |
| eurollm1.7b-base | .1612 | .0534 | .0520 | −.0014 | .0256 → .0211 |
| llama1b-base | .1769 | .0574 | .0532 | −.0041 | .0218 → .0224 |
| eurollm1.7b-ins | .1259 | .0579 | .0535 | −.0043 | .0275 → .0227 |

Per-language CER at 29 h (pt/ro/nl/da/sv): qwen-base .025/.047/.049/.065/.042; qwen-ins .026/.049/.051/.067/.046;
llama-ins .027/.051/.053/.079/.050; eurollm-base .031/.049/.052/.069/.052; llama-base .029/.053/.053/.076/.049;
eurollm-ins .032/.048/.055/.069/.054.

Reading, against §5:

- **The gap closes.** Before the probe the six span .126 to .432; after 29 h per language they span .047 to .054, a
  spread of .006. The pre-probe ranking (EuroLLM first, Qwen last) does not survive: Qwen is first and the two EuroLLM
  arms are fourth and sixth. The OOD-related term in the headline measured what the decoder was pretrained on, not how
  well it learns the language.
- **Decision rule.** The best two arms (qwen-base .0472, qwen-ins .0489) differ by .0017, below the eval-noise band
  (.002 to .010), so the rule sends the choice to trained-language ASR+ST. The only numbers I could read for that are
  the older `02-backbones.md` table on `origin/main`, where qwen-base is ahead of qwen-ins on both `sel (ASR)` (.0858
  against .1067) and `sel (ST)` (.378 against .381); that table predates the EuroLLM IFT results, so re-check against
  the current one before relying on it. On this probe's own trained-language numbers the two are tied (.0168 against
  .0175). The whole six-arm spread is also within the upper end of the band, so the probe separates the backbones
  weakly: it shows that none of them is weak in these languages, not that one is clearly best.
- **Trained-language quality.** No arm lost more than the band (.010) on the five trained languages; the largest
  change is +.0016 (qwen-base); four of six improved. This is measured on FLEURS clips of the trained languages, since
  the probe eval set has no in-domain sets, so it does not replace the `ID` term of the selection score.
- **Slopes.** Every arm is still improving at 29 h. EuroLLM-base is flattest (−.0014); llama-ins is steepest
  (−.0091). The "penalised, not weak" signature (steep slope for an arm behind at 10 h) is clearest for llama-ins; the
  Qwen arms were furthest behind at step 0 and already ahead by 10 h.
- **Caveats.** One seed per arm. da has only 29 h in total, so its 29 h point is its whole pool. CER is clipped at 1.0,
  but no pre- or post-probe per-language CER reached it (the largest is .498), so the clip changed nothing here.

## 6. Not decided here

- Whether the headline `sel` keeps OOD-related. This plan only produces the evidence; the metric stays as
  pre-registered until the PI rules.
- Anything touching `results_master.csv`: the eval-owner writes it. Cells from Steps A and B, if run, go to them as a
  `cells.csv` / `meta.csv` pair under `charts/PROTOCOL.md` §4.

## 7. Order of work

1. Step A is done (above).
2. One Step B arm as a cost and wiring check, then the rest.
3. A scores file next to the others in `selection-metric/scores/`, a §5 note in `02-backbones.md`, a board entry.
