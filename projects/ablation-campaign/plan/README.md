# MELT ablation campaign — research plan

This folder is the working plan for the paper on **what makes a strong,
efficient multilingual speech LLM at 2–3B parameters with late modality
fusion**, and for the two commitments that grew out of it: coverage of the
**24 EU languages** with a per-language "hours → performance" study, and one
large final training run (**Fondue**, `06-fondue.md`).

It is written for two audiences: the people running the campaign, and any
agent picking it up cold. Read it top to bottom once, then keep
`timeline.md` and `00-status.md` open.

## How to read this folder

| file | what it holds |
|---|---|
| `timeline.md` | week-by-week plan with dates, per-track TODO lists, expected outcomes, gates |
| `00-status.md` | living snapshot: what is running, what is blocked, the next decision |
| `01-interface-recipe.md` | step 0 (LibriSpeech) and the five-language interface screen: LR, batch, frame stacking, prompt |
| `02-backbones.md` | the 2×3 backbone grid (Llama 3.2 1B, Qwen3.5 2B, EuroLLM 1.7B; base and instruct) and the text-prior study |
| `03-audio-stack.md` | encoders × adapters, the cost axis, the efficiency figure |
| `04-regime.md` | IFT-level training-regime factors as a half-fraction design |
| `05-language-ladder.md` | the "N hours gives P" study over 24 EU languages, transfer and repetition probes |
| `06-fondue.md` | the big run: data, decisions, freeze date, the Raclette pilot, operations, off-boarding |
| `infrastructure.md` | clusters, repos, data, tooling, and every trap an agent needs to know |
| `board.md` | message board: dated entries from people and agents, newest first |
| `data/` | per-language hours (`hours_by_language.csv`), eval-set inventory, tier table |

Each numbered file opens with a **Settled / Open / Owner** block. Settled items
are decisions already made with the PI; do not re-litigate them in a new
session, post to `board.md` instead if you think one is wrong.

## What the paper argues, and the order the evidence is gathered

1. **The interface must work before anything is compared.** The modality
   alignment (MA) stage as run through August 2026 did not align: eval loss
   sat at 2.6–3.1 nats per token, at the level of an unconditional text LM on
   these transcripts, and generative WER stayed above 1.0. Every comparison
   run on top of that measured "who tolerates a broken adapter best". The
   first work item is therefore a recipe fix on LibriSpeech, then a small
   five-language screen (`01-interface-recipe.md`).
2. **The audio side can be selected at MA-stage cost.** With a working
   recipe, a frozen-encoder / frozen-decoder MA arm already transcribes, so
   MA-stage generative WER is a valid selection metric for encoders, adapters
   and frame rate. An MA arm costs about 30 GPU-h; an IFT arm 80–300. This is
   what makes the audio-stack section affordable (`03-audio-stack.md`).
3. **Backbones are compared on 24 languages, not five**, and explained by a
   text-only prior measured before any training (`02-backbones.md`).
4. **Regime factors** (adapter trainable during IFT, LoRA, ST during MA, MA
   length) are screened once, on the winning backbone, as a half fraction
   (`04-regime.md`).
5. **The language ladder** turns the per-language results into "hours needed
   for a target CER", with family-level transfer estimated from the ladder
   itself and two mixed-tier runs (`05-language-ladder.md`).
6. **Fondue** is trained with the winning architecture and a learning rate set
   by a pilot at 5–10% of its data, and provides the out-of-sample point for
   every curve in 5 (`06-fondue.md`).

## Principles that hold across every section

- **One epoch, derived steps, fixed effective batch.** Arms never pin
  `max_steps`; `batch_duration × grad_accum × world_size` is a campaign
  constant per stage and topology is only a throughput knob. Changing the
  effective batch is a recipe change and forces an LR retune.
- **Noise before claims.** No headline comparison is reported without at
  least two seeds of the baseline at that budget. Seed replicates are cheap
  and the grant expires; they are how spare GPU-hours get spent.
- **Metrics are fixed in advance.** ASR: corpus-level WER and CER after a
  standard multilingual normalizer; CER is the cross-language metric.
  ST: chrF and COMET, BLEU for legacy comparability. Text ability retention
  is measured for every IFT arm. Efficiency: encoder+adapter parameters,
  decoder positions per audio second, training GPU-h per 1,000 audio hours.
- **The evaluation prompt matches the training prompt byte for byte.**
  `melt-eval` enforces this; never hand-write an eval prompt.
- **A run is finished when it has generative scores and a passing exposure
  audit.** Training loss is a health check, never a result.
- **Record what actually ran**: `resolved_config.json`, `arms.tsv` (the
  submission ledger, with SLURM job ids), the exposure audit. Plan files
  quote experiment names, never job ids.

## Decision log

| date | decision | where |
|---|---|---|
| 2026-08-10 | five training languages en/de/fr/es/it at 709 h each, domain-matched to Italian's corpus mix; hold out nl/pt/pl/uk; ST is X→en; MLP is the baseline adapter | campaign README |
| 2026-08-11 | keep the truecase/PNC output as-is (Italian MLS carries ~3% rewrite WER); file the text_field fallback (#66) but do not flip `strict_text_field` yet | training notes |
| 2026-08-23 | every arm runs `num_train_epochs: 1` with derived steps; MA and IFT use `max_duration 60`, `max_tokens 400` | `campaign-one-epoch-convention` |
| 2026-08-24 | DDP for MA arms (4.0× FSDP2); IFT Qwen needs gradient checkpointing under DDP | campaign README |
| 2026-09-02 | one allocation per arm sized to the whole run, not 6 h chunks | campaign.yaml header |
| 2026-09-14 | the August baseline is a failed alignment, not a lower bound; recipe work precedes all comparisons | `01-interface-recipe.md` |
| 2026-09-15 | audio-stack section stays in the paper, screened at MA-stage cost; Q-Former will be fixed by the PI when its week comes | `03-audio-stack.md` |
| 2026-09-15 | EuroLLM-1.7B base and instruct join the backbone grid | `02-backbones.md` |
| 2026-09-15 | Russian and Ukrainian stay in the data; Russian's effect on low-resource Slavic languages is a named probe | `05-language-ladder.md`, `06-fondue.md` |
| 2026-09-15 | the big run is called **Fondue**; its pilot is **Raclette**; config freeze on 2026-10-25 | `06-fondue.md` |
| 2026-09-15 | this `plan/` folder is committed; the repository is being made private | this file |

## Conventions for editing this folder

- Dates are absolute (`2026-10-25`), never "next week".
- Add findings to `board.md` first, then fold the settled part into the
  relevant numbered file. The board is append-only and newest-first.
- Tick TODO boxes in `timeline.md` as work lands; move slipped items forward
  explicitly rather than leaving them unticked in a past week.
- Numbers that came from a measurement say so and name the arm; numbers
  that are extrapolations say "extrapolated".
- Experiment names follow the campaign `EXP_NAME` grammar
  (`projects/ablation-campaign/README.md`). Job ids live in `arms.tsv`.
