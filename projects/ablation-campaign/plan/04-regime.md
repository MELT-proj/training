# 04 — Is modality alignment worth it? Strategy and quantity

**Restructured 2026-10-20 (PI).** The final model is fully fine-tuned, with
no frozen components and no LoRA. The learning rate is Raclette's job
(`06-fondue.md` §4), not this section's. The 2^(4−1) regime fraction
(R1-R10: adapter frozen or trainable at IFT, LoRA, decoder-frozen IFT) and
the MA:IFT ratio sweep (P0-P7) of 2026-09-15/16 are retired. They are in git
history before this date.

**The two questions:**
1. **Is there any measurable benefit from running MA at all,** before a
   full fine-tune?
2. **If so, in which format and how much?** Format means the MA objective:
   gold-transcript CE, self-distillation with ASR and translate
   instructions, or AZeroS's instruction-free SIFT. It also covers ASR-only
   against ASR+ST supervision. Quantity means MA hours.

**What it decides.** The "MA data budget and stage split" row of
`06-fondue.md` §3: whether Fondue runs MA, for how many hours across the
EU languages, and with which objective.

**Owner:** the PI reads the Tier-1 scatter on Mon 10-26 (§5).

## 1. What the self-distillation study already tested

Source: `projects/self-distill/README.md`, merged into main on 10-19.
Corrections to how it is remembered are marked **(note)**.

| phase | stack | what ran | what it shows |
|---|---|---|---|
| 1 | w2v-BERT 2.0 + MLP + Qwen3.5-2B, adapter only, 19.5 h | GOLD (CE), AZEROS, SOFT (forward KL), REVERSE (on-policy reverse KL), RELAY | **Nothing.** No variant aligned, CE included: content-word recall equals a shuffled baseline. Not reusable. |
| 2, start | Whisper-large-v3 + MLP sk5 + Qwen3.5-2B-Instruct, adapter only, en/de/fr/es/it | GOLDW: CE on gold transcripts, 1,000 h, WSD | Aligns at 26-32 h of audio; in-domain WER 0.106 at 1,000 h. This is start **A**. |
| 2, steps 2-4 | same, 50 h **continuations of A**, not runs from scratch | S2 (repeat instruction only = ASR): `ce`, `soft`, `opd`, `opd_anchor`, `sift`, `sift_anchor`. S3 (half repeat, half "translate into X" = ASR + teacher ST): `azeros(_anchor)`, `soft_anchor`, `opd_anchor`. Three data seeds on the main arms. | On-policy never beats offline. The instruction mix gives X→en chrF 24→53 from ASR audio alone, and SIFT ties it on translation and loses on ASR. `S3-azeros_anchor` = start **B**. |
| 2, step 5 | full fine-tune (encoder, adapter, decoder; LRs 6e-6 / 2e-4 / 2e-5) on HR5 and EU24 mixes of gold ASR+ST, 11,499 steps | **(note) Only two starts were fine-tuned, A and B,** not every MA checkpoint. B's runs were cancelled after 2,872-3,590 steps. | B's 13-22 chrF head start is gone by step 718 (383 h nominal), and the last edges by 2,872 (1,532 h; all sign tests p > .2). |
| 2, step 6 | from A, adapter only, 24 EU languages, 2,700 h | SD24 (instruction mix), CE24 (gold CE), SD-SIFT24, each on Qwen and EuroLLM; plus SFT-EU24 on both decoders | Self-distillation *instead of* fine-tuning; none of these starts was fine-tuned afterwards. |

**ASR vs ASR+ST.** It was tested only in this form: S2 against S3, i.e.
repeat-only against repeat + translate instructions with the **teacher's**
translations. It was measured after MA and not carried through fine-tuning,
because only B (ASR + teacher ST) went to step 5. **Gold** ST data inside MA
has never been run.

**Quantity.** It was never varied. Every phase-2 strategy sits on the same
1,000 h CE base, and there is **no no-MA point**: no fine-tune from a random
adapter exists.

## 2. What can be reused

All of it is on the step-5 stack (Whisper-large-v3 + MLP sk5 + Qwen3.5-2B,
adapter-only MA), and every path below exists (checked 2026-10-20).

| piece | MA objective / hours | where | use |
|---|---|---|---|
| GOLDW checkpoints 1200, 2400, …, 9600 | CE, 100-800 h, stable phase of WSD (decay is the last 2,400 steps) | artemis `outputs/GOLDW-…qwen35…-1000h-s42-2g/`, with optimizer and sampler state | **low-hour CE points** after a short decay branch (§3) |
| GOLDW-final = A | CE, 1,000 h | same, plus MN5 | the 1,000 h CE point |
| `SFT-EU24-from-goldw` | A → full FT, all 11,499 steps | nyx `outputs-sft/`, scored at 0/718/…/8616/final | **reused as is** |
| `S3-azeros_anchor` d42 = B | CE 1,000 + SD-mix 50 h | artemis + nyx | the 1,050 h SD-mix point |
| `SFT-EU24-from-sd-azeros` | B → full FT, cancelled at ~3,890 | nyx (718-2,872), MN5 | **reused as is** up to 2,872 |
| SD24, CE24, SD-SIFT24 (Qwen) | CE 1,000 + 2,700 h of SD-mix / CE / SIFT over 24 languages | nyx `outputs-sd24/` (finals), MN5 | the 3,700 h points: they need the fine-tune |
| the same three on EuroLLM, A-EuroLLM, `SFT-EU24-from-EuroLLM-A` | as above | nyx, MN5 | a decoder check, only if EuroLLM wins the backbone |
| S2 arms, other S3 arms | 50 h continuations | artemis | not needed: B stands for them |
| the frozen clone `~/training-sd-sft` @ 2a26f33 on MN5 | the step-5 SFT code | MN5 | **new fine-tunes must run from it**, so that their batches match A's and B's |
| HR5 SFT pair | 5-language fine-tune | nyx | not reusable: different data |

The reuse is large because **only the fine-tune side is missing.**
- Five MA starts exist and have never been fine-tuned.
- Two low-hour CE starts cost a few GPU-minutes.
- The no-MA start costs nothing.

## 3. Design

One downstream protocol for every point: **the step-5 EU24 fine-tune**.
- Same config, seeds and world size as A's and B's runs, from the same clone.
- Every run reads the same batches, so only `model.ckpt` differs.
- Readouts at steps 718, 1,436, 2,154 and 2,872 (383 / 766 / 1,149 /
  1,532 h nominal), plus step 0 for MA alone.
- Arms stop at 2,872. The schedule stays the 11,499-step one, so the LR is
  in its stable phase at every readout for every arm.

**What each tier answers.** Every comparison is made *after* the same
fine-tune, because Fondue fine-tunes everything: MA matters only if what
it gives survives that.
- **Tier 1** reuses what exists and answers three things:
  - *Is MA needed?* No MA (T0) against MA (T1, T2, A, T3) at equal
    fine-tune steps, with T0' and A' for the noise.
  - *Roughly how much?* The same CE objective at 0 / 120 / 360 / 1,000 /
    3,700 h.
  - *Which format?* At equal hours (3,700): ASR only (T3), teacher ST
    (T4), SIFT (T5); A against B at 1,000 h. Gold ST inside MA (T6) was
    dropped on 10-20 (PI: not tested for now).
  
  Its limits: below 1,000 h it covers only the five high-resource
  languages, and every point above 1,000 h stands on the same 1,000 h CE
  base.
- **Tier 2** runs only if Tier 1 says MA matters. It re-measures the
  hours curve the way Fondue would run MA: from scratch, on all 24
  languages, per format. That gives the hours to reserve directly.

**Tier 1 — reuse, submit this week.**

| point | MA objective | MA hours | languages in MA | start | fine-tune |
|---|---|---|---|---|---|
| T0 | none (random adapter) | 0 | — | Whisper + Qwen + a fresh MLP | **new**, to the end (11,499), to compare with A's final |
| T1 | CE | 120 | HR5 | GOLDW ckpt-1200 + 240-step decay | new, to 2,872 |
| T2 | CE | 360 | HR5 | GOLDW ckpt-3600 + 720-step decay | new, to 2,872 |
| A | CE | 1,000 | HR5 | GOLDW-final | **exists** |
| B | CE + SD-mix (ASR + teacher ST) | 1,000 + 50 | HR5 | `S3-azeros_anchor` d42 | **exists** to 2,872 |
| T3 | CE + CE | 1,000 + 2,700 | HR5 then EU24 | CE24 | new, to 2,872 |
| T4 | CE + SD-mix | 1,000 + 2,700 | HR5 then EU24 | SD24 | new, to 2,872 |
| T5 | CE + SIFT | 1,000 + 2,700 | HR5 then EU24 | SD-SIFT24 | new, to 2,872 |
| ~~T6~~ | ~~CE + gold ASR+ST~~ | | | dropped 10-20 (PI) | |
| T0', A' | replicates of T0 and A, data seed changed | | | | new, to 2,872: there is no fine-tune seed noise yet, and the T0-A gap is the headline |

- **Decay branches (T1, T2).** Resume the GOLDW checkpoint and override
  `max_steps` and `num_decay_steps`, so that it decays over the last 20% of
  its own length, as GOLDW did. The old rule "never intermediate
  checkpoints" stands; the branch is what makes them real runs. The
  checkpoints carry optimizer and sampler state, but whether the resume
  takes the overrides has to be checked first. On artemis 2× H200 each
  branch is minutes.
- **T6 dropped (PI, 10-20).** For the record, the gold ST that exists:
  X→en 55,633 h over 21 sources, en→X 2,744 h over 12 targets, none for
  ga/mt. At 2,700 h with a 0.25 ST share that is 24.5 h per direction over
  33 directions.
- **T0's init.** `INIT=none` (PR #159). The model config matches
  GOLDW-final's except `encoder.eval_when_frozen`, which is inert with an
  unfrozen encoder. One caveat: A's start loads through `dtype=auto`, so its
  Whisper weights are F16 rounded to bf16, while T0 keeps the hub's F16.
  That is ~0.1% per weight, below seed noise, but T0 vs A is not purely
  "MA vs none".

**Tier 2 — only if Tier 1 shows an MA effect that survives step 718.**
The Fondue answer then needs the curve over **24 languages** and from
scratch. Below 1,000 h Tier 1 is HR5 only, and above it every point stands
on the 1,000 h HR5 base.
- MA from scratch on the SD24 language mix (flat cap), CE and SD-mix at
  120 / 360 / 1,000 h.
- Six cheap MAs plus six fine-tunes to 2,872.

**Fondue-stack check (beside Tier 1, once the backbone is called).**
- The whole study above runs on Qwen3.5-2B-Instruct + MLP, not on the Fondue
  stack (Whisper + MoE + the winning decoder).
- One extra run tests whether the answer to question 1 carries over: a
  no-MA IFT on the winning backbone with the backbone IFT config,
  `~/training-ift`'s, read against that backbone's existing MA 700 h + IFT
  arm at the same steps. This is the retired P0, kept because it is the
  number Fondue actually inherits.

## 4. Readouts and figures

Per checkpoint, scored on artemis with the step-5 pipeline
(`eval/watch_sft_checkpoints.py`, `render_sft_tables.sh`), plus the
campaign's selection metric:
- FLEURS-24 dev: WER on HR5 and CER on the other 19;
- X→en and en→X chrF++, on HR4 and the other 19;
- the share of en→X outputs in the requested language;
- runaways, i.e. hypotheses over twice the reference length;
- the selection score.

ASR and ST are always reported side by side (rule of 2026-09-20,
SALMONN's task over-fitting): a point that wins on ASR and loses on ST is a
finding, not a winner.

**The figure the PI asked for.**
- x = MA hours (log axis, with 0 shown apart);
- y = each metric after S fine-tune steps;
- one panel per S (718 and 2,872), one marker per strategy;
- a band for the T0/T0' and A/A' seed spread.

**Second view, for the cost argument.** x = total GPU-h (MA + fine-tune).
On this stack an MA hour is not much cheaper than a fine-tune hour:
- CE24: 2,700 h in ~75 GPU-h, i.e. ~36 h per GPU-h;
- SFT-EU24: 6,133 h nominal in ~226 GPU-h, i.e. ~27 h per GPU-h;
- SD-mix: about 2.8× the CE cost per step.

So MA has to buy quality, not just cheap hours.

## 5. Decision rule for Fondue

Read Mon 10-26, at the latest 2,872-step readout available.

| outcome | Fondue |
|---|---|
| T0 is within the seed band of A at 718 and after | **No MA, or the shortest MA that keeps the start stable** (T1's 120 h, if T0 shows loss spikes or a damaged decoder). The MA hours go to IFT. |
| MA helps at 718 but the gap is gone by 2,872 | Same, because Fondue's IFT is far longer than 1,532 h. A short MA is kept only if it buys stability or early checkpoints. |
| A gap persists at 2,872 | MA is kept: the best strategy, at the knee of the hours curve. Tier 2 then sets the EU24 hours. **Fondue's MA can start before Tier 2 ends.** It is WSD, so its length is set when the decay is triggered, not at submission. |
| SD-mix or SIFT beats CE only on ST at step 0 | That repeats step 5, and does not justify the extra cost of self-distillation in Fondue. |

**Default if Tier 1 is late:** Fondue keeps a CE MA on the EU24 pool,
shortened to what the 10-23 freeze already fixes. The decay is triggered
by the first readout that lands.

## 6. Cost and schedule

| item | GPU-h | where | when |
|---|---|---|---|
| T1, T2 decay branches | < 2 | artemis 2× H200 | Tue 10-20 to Wed 10-21 |
| Tier-1 fine-tunes: T1-T5, T0' and A' to 2,872 | 7 × ~60 = ~420 | MN5, 2 nodes each | submitted Wed 10-21 |
| T0 to the end | ~226 | MN5 | submitted Wed 10-21 |
| evals (6-7 checkpoints × 9 runs) | A6000 | artemis | as checkpoints land |
| Fondue-stack no-MA IFT | ~100 (Llama) / ~300 (Qwen) | MN5 | when the backbone is called (~10-22) |
| Tier 2, if triggered | ~500 | MN5 + artemis | 10-26 to ~10-30 |

**Where things run.**
- **artemis `hades`:** the two decay branches only. They must resume
  GOLDW's 2-rank sampler state, so they need GOLDW's world size: 2 H200.
  Queued 2026-10-20 and waiting for GPUs (1 of 8 was free).
- **MN5:** everything that needs a whole node, so that the arms stay
  consistent: all fine-tunes (2 nodes, as A and B) and the Fondue-stack
  IFT. Each is requested with the shortest wall
  time that covers it.

**Checklist to start Tier 1** (checked 2026-10-20 unless marked):

| # | item | state |
|---|---|---|
| 1 | MN5 clone `~/training-ma-utility` = 2a26f33 + the two PR #159 commits (`~/training-sd-sft` stays untouched) | ✓ clean |
| 2 | starts on MN5 scratch: GOLDW-final, `S3-azeros_anchor` d42, CE24, SD24, SD-SIFT24 (Qwen) | ✓ all present |
| 3 | `launch_sft.sh` builds a **no-MA start** (`INIT=none`: no `--model.ckpt`, a fresh MLP, the same processor) | ✓ PR #159 (open), in the clone |
| 4 | the reused starts (CE24, SD24, SD-SIFT24) go through `INIT=sd SD_CKPT=…` with their own `INIT_TAG` | ✓ dry-run table done on MN5 |
| 5 | **a clean stop at step 2,872** without changing the 11,499-step schedule: a stop-at-step option, or a wall time ending just past the 2,872 save. MN5 step time varies up to 1.5×, so the option is safer. | ✓ `run.stop_after_step`, PR #159; `acc_debug` smoke against A's log queued |
| 6 | decay branches T1 and T2: resume GOLDW ckpt-1200 and ckpt-3600 with `max_steps` 1,440 and 4,320 and `num_decay_steps` 240 and 720. At the first logged step, check that the LR starts decaying from 1e-3. It should: the HF WSD scheduler is a LambdaLR rebuilt from the new arguments, with the step count restored from `scheduler.pt`. | queued on `hades`; then the LR check and `mn5transfer` of the two finals (~5 GB each) |
| 7 | ~~T6's MA~~ | dropped 10-20 (PI) |
| 8 | replicates: `SEED=1338` for T0' and A' (it changes the init and the data order together) | launcher supports it |
| 9 | scoring: add the new run names to `runs.txt` for `eval/watch_sft_checkpoints.py` (nyx → artemis A6000), with `pull_mn5_checkpoints.sh` limited to steps 718-2,872 | to do |
| 10 | the selection metric on each readout, beside the step-5 tables | to do |

Items 6 and 9 gate the Wed 10-21 submission, and so does the PI's go for `acc_ehpc` (~630 GPU-h).

The fine-tunes run at ~7 h to step 2,872 after a 12-13 h `acc_ehpc` queue.
Everything submitted on 10-21 is read by Sat 10-24 to Sun 10-25.

## 7. Results

| point | MA objective | MA h | step | WER HR5 | CER other-19 | X→en HR4 / other | en→X HR4 / other | selection | GPU-h |
|---|---|---|---|---|---|---|---|---|---|
| A | CE | 1,000 | 0 / 718 / 2,872 | .081 / .092 / .072 | .582 / .136 / .092 | 25.7 / 22.3, 57.3 / 47.5, 57.7 / 50.1 | | | |
| B | CE + SD-mix | 1,050 | 0 / 718 / 2,872 | .083 / .077 / .072 | .565 / .147 / .098 | 39.2 / 40.3, 57.4 / 47.2, 57.9 / 49.9 | | | |
| T0 … T5 | | | | | | | | | |
