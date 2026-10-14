# Self-distillation, phase 2

Goal: show that on-policy distillation (OPD) from the frozen decoder reading
the transcript improves audio-text alignment beyond gold-transcript CE and
AZeroS-style hard-label distillation (arXiv 2601.06086), for ASR and speech
translation, across languages. The trainer is `melt.training.train_self_distill`
(`melt/training/self_distill.py`, design notes in `docs/self_distillation.md`).

## Where it stands (2026-10-12)

Read this first; the sections below hold the evidence. All numbers are on dev
sets, from the same aligned start, with 50 h of transcription-only data in
en/de/fr/es/it per run.

- **Setup.** Frozen Whisper-large-v3 encoder, a small MLP adapter (the only
  thing that trains), frozen Qwen3.5-2B. The *teacher* is that same decoder
  reading the transcript as text; the *student* is the full model hearing the
  audio. The start, GOLDW-final, is standard training (predict the transcript)
  on 1,000 h: in-domain WER 0.106, and asked to translate it just transcribes.
- **The original hypothesis is not supported.** On-policy distillation (the
  student writes its own answer, the teacher scores it token by token) never
  beat offline imitation of the teacher's answers. With the repeat instruction
  alone it hurts transcription (WER 0.106 -> 0.121, language-model-flavoured
  substitutions); with the instruction mix it translates exactly like offline
  imitation and transcribes worse (0.127 vs 0.112).
- **What holds: translation transfers from transcription data alone.**
  Distilling the decoder's own answers to "repeat" and "translate into X"
  instructions turns the model into a speech translator: X->en chrF 24 -> 53
  on the trained source languages and 21 -> 42 on 19 languages the adapter
  never heard; into de/fr/es/it, and into nl/pt that were never targets, about
  90% of what the decoder scores translating the gold transcript. The output
  is in the requested language 80-97% of the time.
- **Against AZeroS's own recipe** (no instruction anywhere: the student copies
  the decoder's free reply to the bare transcript) translation is a tie, AZeroS
  1-2.5 chrF ahead into de and nl, on three data seeds. AZeroS paraphrases
  instead of transcribing, though: in-domain WER 0.160 / 0.159 / 0.180 against
  0.112 / 0.109 / 0.110. Our mix keeps transcription intact through its repeat
  half, not the gold-CE term, since removing that term changes nothing.
  AZeroS never evaluated speech translation, so the transfer itself is new.
- **Giving AZeroS the transcript as well does not get both.** `sift_anchor`
  (AZeroS's recipe plus predicting the gold transcript under the repeat
  instruction) transcribes well (in-domain WER 0.117) but, asked to translate
  from a trained language, mostly transcribes: X->en chrF 35.8 against 53.5,
  and its output scores 61 chrF against the *source* sentence (other distilled
  arms: ~27). One seed, but the gap is ~18 chrF against ~1 of seed noise. The
  explicit translate instructions are what give translation and transcription
  together, so the self-distilled checkpoint carried forward is
  `S3-azeros_anchor` d42.
- **Next: does the alignment matter after standard fine-tuning?** Step 5
  (below) fine-tunes the whole model on transcription and translation data
  from the standard-aligned start (GOLDW-final) and from `S3-azeros_anchor`,
  on five languages and on all 24 EU languages. Running since 2026-10-12.
- **Open.** Test sets, spoken QA, and the encoder question (Whisper aligned
  after ~30 h; w2v-BERT not even at 3,500 h; no further comparison).

Arm names, by what each trains on:

| arm | the student is trained to |
|---|---|
| `ce` | predict the gold transcript (standard training) |
| `soft` | match the teacher's whole next-token distribution over the teacher's greedy answer (forward KL) |
| `opd` | sample its own answer and match the teacher on it (reverse KL): on-policy distillation |
| `azeros` | reproduce the teacher's greedy answer (cross-entropy): offline imitation |
| `sift` | reproduce the teacher's free reply, with no instruction on either side: AZeroS's recipe |
| `<arm>_anchor` | also predict the gold transcript under the repeat instruction, at weight 0.5 |

`S2-` runs use one fixed prompt (the repeat instruction; none for `sift`, whose
`sift_anchor` keeps the repeat instruction for the gold CE only); `S3-` runs use the instruction mix
(half repeat, half "translate into X", X from en/de/fr/es/it). `-s<seed>-d<data
seed>` names the seed and the data order; runs without `-d` all read data seed
42 (see "Running on artemis").

## Why phase 2 exists

Phase 1 (GOLD / AZEROS / SOFT / REVERSE / RELAY, w2v-BERT 2.0 + Qwen3.5-2B,
1,500 steps at 60 s of audio per step, artemis jobs 335335-335355) compared the
objectives before any of them aligned:

- Every run saw ~19.5 h of audio. The five-language screen
  (`../ablation-campaign/plan/03-audio-stack.md` §1b) shows this w2v-BERT stack
  does not align even at 3,500 h (best mean WER 0.743), while Whisper-large-v3
  with the same recipe reaches WER 0.15-0.23 by ~318 h.
- In-training WER is a fraction (`jiwer`), so GOLD's "1.02" is 102%. Reference
  content words appear in the hypotheses of all five variants no more often than
  in shuffled pairs: no variant read the audio.
- The distilled variants' worse WER is output format: 92% of teacher replies hit
  the 128-token budget, so their targets almost never contained the stop token.

Phase 2 therefore aligns first, with CE on Whisper, and compares objectives as
continuations from aligned checkpoints.

## Plan

1. **Aligned start.** `launch_goldw.sh`: GOLDW, standard MA on Whisper-large-v3 +
   Qwen3.5-2B, 1,000 h, eval every 50 h, a checkpoint every 100 h (also locates
   the alignment transition). `launch_teacher_audit.sh`: what the teacher writes
   for each candidate prompt, per language (reply length, repeat WER, translation
   chrF on FLEURS). Then a melt-eval parity check on a GOLDW checkpoint.
2. **OPD vs CE for ASR**, `launch_step2.sh`, all four arms from GOLDW-final with
   the repeat instruction, 3,000 steps at 30 s x 2 GPUs (50 h), seed 43 (seed 42
   would replay GOLDW's first 50 h): `ce` gold CE continued; `soft` forward KL on
   the teacher's greedy repeat (= the transcript, so this is the soft-label
   version of AZeroS); `opd` reverse KL on student samples, teacher = the frozen
   decoder reading the transcript under the student's instruction
   (`teacher_prompt=mirror`); `opd_anchor` = `opd` + 0.5 x gold CE. Scored by
   `eval/launch_eval_step2.sh` (in-domain WER per source, zero-shot WER on 19
   unseen FLEURS languages, zero-shot X->en speech translation with the
   teacher audit's translate instruction) and `alignment_probe.py`.
3. **Task-conditioned multilingual OPD**: an instruction mix (repeat, or
   translate into one of en/de/fr/es/it), targets from the teacher, against the
   same mix trained offline at equal compute.
4. **Claim hardening** (chosen after step 3; spoken QA left for later):
   - *Seeds*: `azeros_anchor` and `opd_anchor` at seeds 44 and 45, with the
     data order varied too (`-s44-d44`). Every earlier S2/S3 run, whatever its
     `-s<seed>`, read data seed 42 (see "Running on artemis"): the first `-s44`
     runs replicate `-s43` on the same batches and measure only numerical and
     sampling noise.
   - *AZeroS's own recipe* as the head-to-head: `ARM=sift`, the azeros loss with
     no instruction on either path (teacher = the decoder replying to the bare
     transcript, student = `{audio_token}` alone), no anchor, no mix, same warm
     start, data and steps, seeds 43-45. The paper (Qwen2.5-7B-Instruct
     teacher, projector-only training) gives no decoding settings or reply cap
     and evaluates no speech translation; here the teacher decodes greedily
     under the 448-token budget every arm has. Plus `azeros` with the mix but
     without the anchor, so a gap to SIFT cannot be the gold CE's, and
     `sift_anchor` (SIFT plus the gold CE under the repeat instruction,
     `distill.student_template`), which gives SIFT the word-for-word
     transcription signal the mix has in its repeat half.
   - *Translation into other languages*: FLEURS X->Y dev sets for de/fr/es/it
     (trained targets) and nl/pt (targets never trained), 23 sources x 100 each
     (melt-eval branch `claude/fleurs-x-to-y-st`), read against
     `text_ceiling.py`: the frozen decoder translating the gold transcript under
     the same instruction.

## Results so far

**Teacher audit** (`teacher_audit.py`, Qwen3.5-2B text-only, 210 validation
utterances per language from CV22/MLS/VoxPopuli, FLEURS dev for translation;
jobs 335829 and 335832, outputs under `.../outputs/sd-teacher-audit/`):

| teacher prompt | result |
|---|---|
| bare transcript (phase 1) | median reply 200-430 tokens, 73-92% longer than 128; restates the transcript 4-10% of the time, otherwise discusses it |
| `{content} Repeat the above content.` (AZeroS SIT_s) | WER 0.20-0.45; short questions and commands get answered, 2-9% refusals |
| `{content} Repeat ... exactly, word for word, and write nothing else.` | WER 0.05-0.18, almost all of it the instruction echoed after the transcript |
| `Repeat the following content exactly, word for word, and write nothing else.\n\n{content}` | **WER 0.000, exact match 99.5-100% in all five languages** |
| `Content:\n{content}\n\nRepeat the content exactly, ...` | WER <= 0.004 |
| `Translate the following content into {target}, and write nothing else.\n\n{content}` | chrF 59-65 into English, 47-58 between the other four |

The instruction has to come before the content: in one user turn Qwen3.5-2B
reads a trailing instruction as part of the content. Phase 2 uses the
instruction-first templates for both the teacher (transcript) and the student
(`{audio_token}`). With them the teacher's greedy repeat *is* the transcript, so
for ASR an offline teacher-target arm is gold CE; the ASR arms separate hard vs
soft labels and off- vs on-policy instead.

**GOLDW** (artemis job 335828, 2x H200, 5 h 13 m). Trainer eval, 50 clips per
language, the same clips on which phase-1 GOLD scored 1.0-3.0:

| step (audio) | en | de | fr | es | it |
|---|---|---|---|---|---|
| 600 (50 h) | 0.128 | 0.167 | 0.172 | 0.132 | 0.160 |
| 1,200 (100 h) | 0.108 | 0.147 | 0.471 | 0.115 | 0.138 |
| 6,000 (500 h) | 0.091 | 0.118 | 0.132 | 0.077 | 0.104 |
| 12,000 (1,000 h) | 0.083 | 0.102 | 0.134 | 0.065 | 0.092 |

Training loss sat on the phase-1 plateau (~3.0) until ~26 h of audio and had
crossed by ~32 h. The French spikes (1,200 and 2,400) are one runaway clip of
50 each time: insertion rate 0.34 and length ratio 1.64 at step 1,200, with the
eval loss still falling.

**Alignment probe** (`alignment_probe.py`: audio-vs-text retrieval within one
corpus, and per-token JSD between the decoder's next-token distributions given
the audio and given the transcript, teacher-forced on the gold transcript):
phase-1 GOLD scores R@1 0.03-0.07 at its best layer (chance 0.03) and JSD
0.43-0.48 per token in all five languages; GOLDW-final scores R@1 0.96-0.99 and
JSD 0.10-0.13. The probe separates an aligned model from an unaligned one.

**GOLDW-final on the step-2 battery** (`eval/launch_eval_step2.sh`, A6000,
melt-eval, scored by `eval/summarize_evals.py`; means over languages):

| prompt | in-domain WER (4,500 utts) | FLEURS WER, 5 trained | FLEURS WER / CER, 19 unseen | X->en chrF, trained / unseen | chrF vs source, trained |
|---|---|---|---|---|---|
| bare (as trained) | 0.108 | 0.081 | 1.28 / 0.75 | 24.4 / 21.4 | 86.5 |
| repeat instruction | 0.106 | 0.082 | 1.20 / 0.68 | | |

The repeat instruction the step-2 arms train with costs GOLDW nothing. Asked
to translate, GOLDW transcribes: its output is the source transcript (chrF
77-91 against it), and the 23-25 chrF against English is what shared names,
numbers and cognates give a copy. Unseen languages come out in the nearest
trained one, sometimes with the meaning partly carried over: Portuguese as
Spanish-like text, Polish as German ("... die planen eine jährliche ... wächst
die popularität" for "An increasingly more popular option for those planning a
gap-year ..."). Up to 21% of utterances in an unseen language run away.

**Step 2: continuations under the repeat instruction** (all from GOLDW-final,
seed 43, 50 h; battery on A6000; probe under the repeat instruction with the
mirror teacher, teacher-forced on gold, mean over five languages):

| arm | in-domain WER | FLEURS WER, 5 trained | FLEURS CER, 19 unseen | X->en chrF trained / unseen | chrF vs source | NLL(gold given audio) | KL(audio \|\| text) |
|---|---|---|---|---|---|---|---|
| GOLDW-final (repeat prompt) | 0.106 | 0.082 | 0.684 | 24.4 / 21.4 | 86.5 | 0.348 | 1.341 |
| `ce` | 0.108 | 0.081 | 0.713 | 24.3 / 21.2 | 86.4 | 0.345 | 1.365 |
| `opd` | 0.121 | 0.083 | 0.994 | 24.3 / 22.4 | 86.2 | 0.373 | 1.253 |
| `opd_anchor` | 0.116 | 0.082 | 0.935 | 24.3 / 21.5 | 86.3 | 0.362 | 1.259 |

Continued CE is a null on every metric (its in-training 50-clip WER gain, 0.095
-> 0.090, does not survive 4,500 utterances: read no trend into the 50-clip
evals). When the teacher's target is the gold transcript, OPD is worse than CE
everywhere and transfers nothing to translation. It does what reverse KL
promises -- less student mass where the teacher has none (KL(audio || text)
-8%) -- and pays in gold likelihood (+7%), which is what WER measures; the
probe's JSD improves meanwhile, so JSD alone is not an alignment metric. The
errors are systematic, language-model-flavoured substitutions ("in schweiß
gebadet" -> "getränkt", dialect normalised, "duemilatrenta" -> "trecento e
trenta"). The teacher is not their source: on gold prefixes its NLL is
0.001-0.011 nats/token, i.e. one-hot. The working explanation is the student's
own deviated prefixes, on which the teacher can only continue fluently, so the
student learns to follow its prefix over the audio. The gold anchor (0.5 x CE)
removes about a third of the damage. On unseen languages the OPD arms run away
more (worst language 25-27% of utterances vs 19%).

**Step 3: the instruction mix** (`TRANSLATE_FRAC=0.5`: half the utterances
are distilled under "Translate the following content into {tgt_lang}, and
write nothing else." with a target drawn from en/de/fr/es/it minus the source;
the teacher reads the same instruction over the transcript; both arms keep the
0.5 x gold-CE anchor on the repeat prompt; from GOLDW-final, seed 43, 50 h):

| arm | in-domain WER | FLEURS CER, 19 unseen | X->en chrF / BLEU, 4 trained sources | X->en chrF / BLEU, 19 unseen sources | chrF vs source | NLL(gold given audio) |
|---|---|---|---|---|---|---|
| GOLDW-final | 0.106 | 0.684 | 24.4 / 0.1 | 21.4 / 0.9 | 86.5 | 0.348 |
| `opd_anchor` (on-policy, reverse KL) | 0.127 | 0.948 | 53.4 / 22.7 | 42.3 / 14.1 | 26.4 | 0.373 |
| `azeros_anchor` (offline, CE on the teacher's greedy reply) | **0.112** | **0.692** | **53.5 / 23.7** | **42.8 / 14.9** | 27.7 | 0.356 |

From ASR data alone, both arms turn the speech LLM into a speech translator:
X->en chrF 53 on the trained source languages (the text-only teacher reading
the gold transcript scores 59-65) and 42 on 19 source languages the adapter
never heard (Portuguese 60, Swedish 55, Danish 52-53, Polish 47). On-policy
training adds nothing to it -- per-language chrF agrees within 1-2 points --
and costs ASR: offline matches the translation at a third of the in-domain
cost and keeps unseen-language ASR intact. The translation ability is already
there at 1,000 steps (on-policy: 52.8 / 42.1). Seeds, an offline soft-label
arm and AZeroS's own recipe are in step 4.

**Step 4: claim hardening.** Same start, data budget and battery, plus the
FLEURS X->Y sets and the text-only ceiling. `d42`/`d44` is the data seed. Unseen
means here leave out Irish (ga), which has no source transcript for the ceiling:
18 unseen sources, so X->en reads 44.3 where the step-3 table reads 42.8.

| arm | in-domain WER (without runaway loops) | FLEURS CER, 19 unseen | X->en chrF trained / 18 unseen | NLL(gold given audio) |
|---|---|---|---|---|
| GOLDW-final | 0.106 (0.102) | 0.684 | 24.4 / 21.8 | 0.348 |
| `S3-azeros_anchor` d42 / d44 | 0.112 / 0.109 (0.107 / 0.105) | 0.692 / 0.692 | 53.5 / 44.3, 52.6 / 43.6 | 0.356 / 0.355 |
| `S3-azeros` (no anchor) d42 | 0.113 (0.107) | 0.697 | 53.7 / 44.4 | 0.363 |
| `S3-soft_anchor` d42 | 0.111 (0.107) | 0.760 | 53.4 / 43.4 | 0.357 |
| `S3-opd_anchor` d42 | 0.127 (0.111) | 0.948 | 53.4 / 43.7 | 0.373 |
| `S2-sift` (AZeroS) d42 / d44 | 0.160 / 0.159 (0.144 / 0.143) | 0.631 / 0.643 | 53.6 / 43.5, 53.7 / 43.5 | 0.486 / 0.482 |
| `S2-sift_anchor` d42 | 0.117 (0.109) | 0.810 | 35.8 / 38.2 | - |
| text-only ceiling | - | - | 61.4 / 56.6 | - |

All three data seeds (42 / 44 / 45), same columns:

| arm | in-domain WER | FLEURS CER, 19 unseen | X->en chrF trained / 18 unseen |
|---|---|---|---|
| `S3-azeros_anchor` | 0.112 / 0.109 / 0.110 | 0.692 / 0.692 / 0.696 | 53.5 / 44.3, 52.6 / 43.6, 53.2 / 43.9 |
| `S3-opd_anchor` | 0.127 / 0.120 / 0.118 | 0.948 / 0.956 / 0.980 | 53.4 / 43.7, 53.2 / 43.5, 53.6 / 43.5 |
| `S2-sift` | 0.160 / 0.159 / 0.180 | 0.631 / 0.643 / 0.666 | 53.6 / 43.5, 53.7 / 43.5, 53.9 / 43.5 |

X->Y chrF, trained sources / 18 unseen sources (nl and pt were never
translation targets in training):

| arm | ->en | ->de | ->fr | ->es | ->it | ->nl | ->pt |
|---|---|---|---|---|---|---|---|
| GOLDW-final | 24.4/21.8 | 20.4/24.4 | 25.7/26.6 | 25.6/27.8 | 24.9/28.0 | 22.7/24.5 | 26.2/27.2 |
| `S3-azeros_anchor` d42 | 53.5/44.3 | 45.2/38.8 | 50.8/40.2 | 45.4/36.3 | 45.8/36.9 | 41.6/35.8 | 50.9/38.1 |
| `S3-azeros_anchor` d44 | 52.6/43.6 | 45.0/38.7 | 50.6/39.8 | 44.9/36.3 | 45.4/36.6 | 40.2/35.9 | 50.6/37.9 |
| `S3-opd_anchor` d42 | 53.4/43.7 | 46.4/37.8 | 50.4/39.3 | 45.6/35.9 | 46.7/36.8 | 41.8/35.0 | 50.8/37.4 |
| `S2-sift` d42 | 53.6/43.5 | 46.8/38.7 | 51.0/40.4 | 45.4/36.3 | 46.2/37.1 | 42.6/36.5 | 50.8/37.7 |
| `S2-sift` d44 | 53.7/43.5 | 46.7/38.8 | 50.8/40.5 | 45.8/36.2 | 46.5/37.1 | 42.7/36.8 | 50.9/37.6 |
| `S2-sift_anchor` d42 | 35.8/38.2 | 38.3/36.5 | 43.6/38.7 | 41.8/35.0 | 41.4/35.6 | 33.0/34.5 | 47.6/36.6 |
| text-only ceiling | 61.4/56.6 | 50.9/44.7 | 55.0/47.4 | 50.1/41.0 | 49.8/43.0 | 47.3/39.1 | 55.3/43.3 |

- **Translation into other languages works, and the instruction generalises.**
  Every distilled arm reaches about 90% of the ceiling on the trained sources,
  for targets it never trained on (nl, pt) as much as for those it did. Which
  language the output is in, read as the FLEURS translation of the same
  sentence it matches best: the requested one 80-97% of the time for every
  distilled arm, SIFT included. GOLDW, asked to translate, copies the source
  91-98% of the time on the trained sources.
- **AZeroS's recipe (`sift`) ties on translation and loses on transcription.**
  Its in-domain errors are paraphrase ("ennuiant" for "angoissant") and a few
  repetition loops; without the loops its WER is still 0.144. On unseen
  languages it often answers the repeat instruction with a translation into a
  trained language, which lowers CER without being transcription. Its teacher
  replies are long (200-365 tokens per batch on average in the first logged
  steps, 14-50% of them cut at 448), and it is generation-bound: 12 s/step on
  H200, 44 s/step on the H100 node.
- **The anchor is not what keeps transcription intact.** The mix without it
  (`S3-azeros`) matches the anchored arm everywhere; the repeat half of the mix,
  whose teacher answer is the transcript, does that job.
- **On-policy (`opd_anchor`) still adds nothing.** Its extra in-domain WER is
  mostly repetition loops (0.111 without them), and unseen-language CER rises
  to 0.948. Soft labels (`soft_anchor`) add nothing over hard ones.
- **Seed noise.** Changing the data seed moves translation by up to 1.4 chrF
  (mostly under 0.5) and in-domain WER by 0.003 (0.02 for SIFT, whose third
  seed paraphrases more). The first `-s44` runs (same data as `-s43`) agree with
  `-s43` within 0.1-0.4 chrF, which is numerical and sampling noise only.
- **`sift_anchor` copies instead of translating.** The gold transcript under
  the repeat instruction fixes SIFT's transcription (0.160 -> 0.117) but pulls
  the translate requests toward transcription: on trained sources its X->en
  output scores 61.0 chrF against the source sentence (SIFT 26.5, our mix
  27.7), and every X->Y target loses 3-10 chrF against the other arms. Unseen
  sources lose less (38.2 vs 43.5), and unseen-language CER rises to 0.810.

**Parity** (melt-eval vs the trainer's own eval, GOLDW-final, the trainer's 250
clips from `eval_parity_spec.py`): WER en 0.081 / 0.083, de 0.100 / 0.102, fr
0.134 / 0.134, es 0.070 / 0.065, it 0.085 / 0.092 (melt-eval / trainer); 37 of
the 50 logged hypotheses identical, the rest differing as bf16 batch padding
would. melt-eval's scorer and the trainer's normaliser agree exactly.

**melt-eval on artemis** imports `melt` from a copy frozen at 7be90c7
(2026-09-08) that predates `stack_factor` and Whisper windowing; it cannot load
any phase-1 or phase-2 checkpoint (`fc1` shape mismatch). Run it with
`PYTHONPATH` pointing at the checkout that trained the checkpoint.
`eval_parity_spec.py` pins melt-eval to the trainer's exact eval cuts (checked on
phase-1 GOLD: 50/50 logged references reproduced).

## Step 5: standard fine-tuning on top (stopped 2026-10-13)

**Stopped by decision on 2026-10-13**, after the step-2872 tables below: the
self-distilled start's head start is gone by then, so the study moved to using
self-distillation *instead of* the fine-tuning step (step 6). The HR5 pair and the
EU24 run from the self-distilled start were cancelled (their outputs are kept); the
EU24 run from the standard start (`SFT-EU24-from-goldw`) ran to its end and is the
full-fine-tuning reference of step 6.

Question: if the whole model is then fine-tuned on ordinary transcription and
translation data, is it better to start from the standard-aligned checkpoint
or from the self-distilled one? The two starts differ only in the adapter:
**A** = GOLDW-final, **B** = `S3-azeros_anchor` d42.

- **Design.** 2 starts x 2 data conditions, one seed (1337 on the command
  line). Within a condition the two runs share config, seeds and world size,
  so they see the same batches; only `model.ckpt` differs.
  - HR5: ASR en/de/fr/es/it, X->en and en->X for de/fr/es/it, up to 600 h per
    entry (an entry is one ASR language or one translation direction).
  - EU24: ASR for all 24 EU languages and every X->en / en->X direction with
    data, under a common cap (197.7 h) solved so the total matches HR5's.
  - Both 6,132.6 h, one pass at 1,920 s of audio per optimizer step: 11,499
    steps (batches come out ~85% full, so ~5,200 h are actually seen).
- **Recipe.** Encoder, adapter and decoder all train (LRs 6e-6 / 2e-4 / 2e-5,
  from SFT-v1.4.0), warmup-stable-decay with 5% warmup and a cosine decay over
  the last fifth to 0.1x. Instructions are drawn at random from
  `TASK_TEMPLATES`; the in-training validation pins one template per task.
- **Precision.** Both starts load as bf16 (transformers 5 loads `model.ckpt`
  with dtype="auto") and DDP does not upcast, so the first smoke test trained
  in pure bf16: AdamW updates below half a bf16 ulp are lost, which at 2e-5
  freezes every decoder weight above ~0.008 and nearly the whole encoder at
  6e-6. `--model.master_dtype float32` (0e19639) keeps fp32 master weights;
  compute stays bf16. A checkpoint is 29 GB (fp32 weights and Adam state).
  The warm starts still load through bf16 before the cast, so A and B start
  from identical encoder and decoder weights and differ only in the adapter.
- **Memory.** fp32 masters peak at 62.7 GiB per GPU at 60 s per micro-batch
  (H200). That does not fit MN5's 64 GB H100s (EU24 batches carry more text
  tokens). `--trainer.ddp_gradient_as_bucket_view true` (2e874c7, 2a26f33)
  shares the gradients with DDP's buckets and zeroes them in place (Trainer's
  `model.zero_grad()` would free them and the accumulation would allocate a
  second copy): 49-54 GiB on MN5, same arithmetic.
- **Files.** `sft/build_sft_mix.py` renders `sft/SFT-HR5.yaml` and
  `sft/SFT-EU24.yaml` from `sft/SFT-template.yaml`; `sft/launch_sft.sh` derives
  the schedule (16 saves, 32 in-training evals) and submits. Evaluation uses
  `eval/format-sft-random.yaml` (random templates, seeded per sample) on FLEURS
  ASR (24 languages), X->en and en->X (23 each). SFT trains on lowercased
  targets, so translation must be scored case-insensitively.
- **Where.** HR5 pair on the artemis H200 node, 2 GPUs each with 16-step
  accumulation, ~19.5 s/step since 2026-10-12 14:20 UTC; each resumes once
  after the 2-day limit (dependent jobs already queued). EU24 pair on MN5, 2
  nodes each with 4-step accumulation, ~9.6 s/step since 2026-10-13 06:10 UTC.
  Outputs under `/mnt/scratch-nyx/giuseppe/melt/outputs-sft/`.
- **Scoring.** `eval/watch_sft_checkpoints.py` runs on nyx and scores each
  scheduled checkpoint on artemis A6000s as it lands (`eval/launch_eval_sft.sh`,
  bf16 as at step 0); `eval/pull_mn5_checkpoints.sh` copies the EU24 weights
  over; `eval/render_sft_tables.sh` writes the tables to
  `outputs-sft/scoring/summaries/tables.md`.

The two starts at step 0 and after 718 steps of HR5 (FLEURS dev, 100 per
language or direction, translation scored case-insensitively; HR =
en/de/fr/es/it, "other 19" = the remaining EU languages, Irish included):

| | step 0 A | step 0 B | 718 A | 718 B |
|---|---|---|---|---|
| ASR WER, 5 HR languages | 0.081 | 0.083 | 0.072 | 0.074 |
| ASR CER, other 19 | 0.582 | 0.565 | 0.536 | 0.563 |
| X->en chrF, HR / other 19 | 25.7 / 22.3 | 39.2 / 40.3 | 57.2 / 46.5 | 57.3 / 46.9 |
| en->X chrF, HR / other 19 | 28.9 / 20.0 | 51.2 / 39.0 | 49.9 / 32.1 | 50.7 / 33.2 |
| en->X output in the requested language, HR / other | 21% / 11% | 90% / 82% | 94% / 69% | 95% / 71% |

- **The head start is gone within ~380 nominal hours, in both conditions.**
  B's 13-22 chrF advantage in translation shrinks to 0.1-1.1 chrF after 718
  steps. A small edge lingers for a while (HR5 at 718: B better into 18 of 23
  en->X targets, p = .01; EU24 at 1436-2154: B better on X->en, p = .03 and
  .01) and is gone by step 2872, where every sign test has p > .2.
- **At step 2872 the data condition decides, not the start** (A / B):

  | | HR5 | EU24 |
  |---|---|---|
  | ASR WER, 5 HR languages | 0.071 / 0.067 | 0.072 / 0.072 |
  | ASR CER, other 19 | 0.770 / 0.783 | 0.092 / 0.098 |
  | X->en chrF, HR / other 19 | 58.7 / 46.6 vs 58.3 / 46.5 | 57.7 / 50.1 vs 57.9 / 49.9 |
  | en->X chrF, HR / other 19 | 51.7 / 19.3 vs 51.6 / 19.0 | 51.7 / 41.2 vs 51.8 / 40.6 |
  | en->X in the requested language, other 19 | 7% / 6% | 93% / 92% |

  Fine-tuning on five languages erodes everything without data, for both
  starts: translation into untrained targets collapses into the trained
  languages (Greek falls from B's 36 chrF at step 0 to under 1) and
  unseen-language ASR degrades. Training on all 24 holds them.
- The fine-tuned models write lowercase (the loader lowercases targets) and
  Europarl-style tokenised spacing ("l ' archipelago"), which hurts BLEU, not
  chrF. Under random templates B's step-0 X->en sits well below its 53.5 under
  the repeat-first format of step 4. The en->X targets do not share their
  English recordings, so compare A and B within a target.

## Step 6: self-distillation instead of fine-tuning, over 24 languages (SD24)

Question (2026-10-13): if the head start of step 5 does not survive fine-tuning, can
modality alignment with self-distillation itself replace the fine-tuning step? Compared,
all from the same standard-aligned start **A** (Whisper-large-v3, MLP adapter,
Qwen3.5-2B):

| arm | what trains | data / labels |
|---|---|---|
| A | adapter | 1,000 h of 5-language ASR, gold transcripts (the start) |
| SFT-EU24 | encoder, adapter, decoder | 6,133 h, gold ASR and ST labels (step 5, from A) |
| **SD24** | adapter | 2,700 h of ASR audio in 24 languages, offline instruction-mixed self-distillation |
| CE24 (control) | adapter | the same audio in the same batches, gold-transcript CE, no teacher |
| **SD-SIFT24** | adapter | the same audio, AZeroS's recipe (step 4's `sift`): no instruction on either path, the student copies the decoder's free reply to the bare transcript |

and the SD24 and SD-SIFT24 recipes again with `utter-project/EuroLLM-1.7B-Instruct-2512` as the decoder.

- **Data.** `sft/SD-EU24-2700h.yaml`, rendered by `sft/build_sft_mix.py --sd24-hours 2700`
  with the same audited sources as the SFT mixes, ASR entries only (the translate rows are
  made from the same audio). Nine languages (de, en, es, fr, hu, it, nl, pl, pt) draw 201.7 h
  each, the other 15 use all their data once (fi 193.7, cs 160.3, ro 130.6, sv 110.3, sk 54.2,
  bg 35.2, lv 30.5, da 29.0, lt 28.4, el 27.0, et 24.3, hr 21.2, sl 15.3, ga 12.7, mt 12.3 h):
  a flat cap, not the beta=0.5 temperature of `config/train/SFT-v1.4.0.yaml`, which would give
  English 34% and Irish and Maltese ~9 h each. The cap equals SFT-EU24's per-language ASR
  audio (197.7 h), so the arms differ in method, not in language balance. Cuts up to 30 s.
- **Recipe.** `sft/launch_sd24.sh` (wraps `launch_step2.sh`). Offline CE on the frozen
  decoder's greedy reply to "repeat" or "translate into X" (translate share 0.25; targets the
  22 EU languages without ga/mt for Qwen, which cannot translate into them: chrF 19 / 27 and
  25% / 19% runaway replies), no gold anchor. Adapter LR 2e-4, WSD, 8 H100 x 64 s per step,
  at most 16 cuts per micro-batch (batches carry ~80% of the budget), fp32 master weights
  for the adapter only (`model.master_dtype_scope: trainable`; the frozen backbones stay
  bf16), teacher replies capped at min(256, 3.5 x content + 32) tokens with the cut ones
  dropped (`distill.reply_budget_factor`, `distill.drop_truncated`).
- **Only half of the audio is transcription.** The repeat rows are the ASR rows: at a translate
  share of 0.25 SD24 has ~2,000 h of them against SFT-EU24's 2,665 h of gold ASR audio.
- **SD-SIFT24** (added 2026-10-14): `ARM=sift sft/launch_sd24.sh`, the same data, batch, schedule and
  adapter-only setting as SD24, with AZeroS's targets instead of the instruction mix: the teacher is the
  decoder replying greedily to the bare transcript (`distill.teacher_prompt=bare`, chat template, no
  instruction), the student hears `{audio_token}` alone, no mix, no anchor. A free reply is a discussion,
  not a copy of the content, so the content-scaled budget and the drop of cut replies are off: every reply
  runs to the 256-token cap or stops on its own, and a cut one stays in the loss. Names: `SD-SIFT24-from-<init>-...`.
  **It is generation-bound**: the MN5 smoke test measured 8.07 s/step at a cap of 256 against SD24's
  3.95 (replies average ~190-250 tokens and 48-94% of the rows of a micro-batch are cut at the cap, so every
  micro-batch decodes the full 256 tokens), i.e. ~53 h and ~430 GPU-h per run. `distill.rollout_lookahead`
  (`ROLLOUT_LOOKAHEAD`) decodes the replies of several micro-batches in one `generate` call: greedy decoding is
  limited by the latency of a token, not by the number of rows. With 4 the same smoke test ran at
  **3.0 s/step** (~20 h, ~165 GPU-h per run); the replies match the plain run step by step (mean reply length
  223.3 vs 223.2 tokens over the first 40 steps, mean |difference| 2.2 tokens; cut share 0.73 vs 0.72) and the
  peak memory is the same (60.7 of 65.2 GB, the activations and the 248k-vocabulary logits of full-length replies:
  SD24 peaks at 20-31 GB). The lookahead is off by default and does not change the SD24 runs.
- **Teacher ceilings** (`text_ceiling.py`, FLEURS dev, 100 per language: the frozen decoder
  reading the gold transcript; chrF, and the share of clips it copies exactly under the repeat
  instruction). Qwen3.5-2B copies 99.5%, so for it the repeat half is gold in effect and the
  anchor is redundant. EuroLLM-1.7B-Instruct translates much better into the small languages
  but does not copy: it paraphrases, and the 2512 version adds explanations or refuses in ~13%
  of clips. Its repeat rows therefore take the gold transcript (`distill.repeat_from_gold`) and
  only the translate rows are the teacher's. The public EuroLLM-1.7B-Instruct scores a little
  above the 2512 on translation.

  | lang | X->en chrF: Qwen / Euro-1.7B / Euro-2512 | en->X chrF: Qwen / Euro-1.7B / Euro-2512 | exact copies %: Qwen / Euro-1.7B / Euro-2512 |
  |---|---|---|---|
  | bg | 61.7 / 65.6 / 64.0 | 52.0 / 62.9 / 54.8 | 100 / 10 / 33 |
  | cs | 60.3 / 65.2 / 63.2 | 47.5 / 55.5 / 52.9 | 99 / 10 / 34 |
  | da | 66.2 / 69.5 / 67.8 | 56.4 / 64.8 / 63.5 | 100 / 9 / 28 |
  | de | 63.6 / 66.8 / 65.5 | 56.4 / 60.3 / 60.2 | 100 / 9 / 46 |
  | el | 57.8 / 59.5 / 60.7 | 41.7 / 47.9 / 44.9 | 100 / 42 / 61 |
  | es | 58.5 / 58.6 / 57.8 | 52.7 / 55.4 / 54.4 | 100 / 22 / 38 |
  | et | 47.9 / 58.6 / 49.5 | 34.7 / 56.3 / 51.7 | 99 / 14 / 52 |
  | fi | 50.8 / 55.6 / 50.6 | 41.8 / 54.0 / 49.0 | 99 / 27 / 40 |
  | fr | 63.4 / 65.5 / 64.5 | 62.2 / 67.9 / 65.2 | 100 / 32 / 61 |
  | ga | - / - / - | 19.3 / 40.7 / 37.7 | 96 / 9 / 1 |
  | hr | 59.3 / 60.9 / 53.7 | 50.7 / 53.3 / 43.4 | 100 / 3 / 35 |
  | hu | 52.9 / 54.8 / 53.3 | 40.9 / 49.6 / 47.4 | 99 / 17 / 55 |
  | it | 60.4 / 62.8 / 61.4 | 54.0 / 57.8 / 56.2 | 100 / 28 / 46 |
  | lt | 51.0 / 55.3 / 50.9 | 40.6 / 52.8 / 49.2 | 100 / 29 / 36 |
  | lv | 50.1 / 56.6 / 49.9 | 35.4 / 50.7 / 48.3 | 100 / 12 / 45 |
  | mt | 54.6 / 57.2 / 61.2 | 26.5 / 56.1 / 49.4 | 99 / 14 / 44 |
  | nl | 55.6 / 58.3 / 57.6 | 51.9 / 54.8 / 53.5 | 100 / 37 / 60 |
  | pl | 54.0 / 57.2 / 55.7 | 44.1 / 50.5 / 48.1 | 99 / 8 / 43 |
  | pt | 64.5 / 64.0 / 62.0 | 61.1 / 66.7 / 63.7 | 100 / 28 / 54 |
  | ro | 62.3 / 66.8 / 65.2 | 53.7 / 62.5 / 60.1 | 99 / 6 / 44 |
  | sk | 56.5 / 64.9 / 61.2 | 45.7 / 58.5 / 55.1 | 100 / 6 / 32 |
  | sl | 54.3 / 59.2 / 55.6 | 44.5 / 55.2 / 51.7 | 99 / 7 / 27 |
  | sv | 62.9 / 68.7 / 67.3 | 56.0 / 65.3 / 61.3 | 100 / 31 / 52 |
  | mean | 57.7 / 61.4 / 59.0 | 46.5 / 56.5 / 53.1 | 99.5 / 20 / 43 |

  (X->en: 22 languages, no Irish source text in FLEURS. Mean en->X over the ten lower-resource
  targets bg, el, et, ga, hr, lt, lv, mt, sk, sl: 39.1 / 53.4 / 48.6.)
- **What the MN5 smoke tests found** (2 nodes x 4 H100, 64 GB; `acc_debug`, 60 steps):
  120 s per rank ran out of memory with the gold anchor (a second graph over the batch) and
  without it. What is large is the student's logits, padded to rows x longest reply x a 248k
  vocabulary and upcast to fp32: 4.76 GiB for 20 rows of 257 tokens, which one looping reply in
  a batch of short utterances produces. Cuts over 28 s (the top 10% of the mix, up to 60 s)
  also ran every rank's decoding to the 448-token cap. With the cap, the reply budget, the row
  limit and bf16 backbones: **4.4 s/step, 20-31 GB per GPU**, 1.3% of rows dropped as cut. The
  control runs at ~1.5 s/step. The dataloader needs ~20 min to deliver its first batch on MN5.
- **Runs** (2026-10-14, MN5 `acc_ehpc`, 2 nodes each, from the frozen clone
  `~/training-sd24-qwen` @ 07f84c2): SD24 Qwen `47281435` (~29 h, ~232 GPU-h) and CE24 control
  `47281436` (~10 h, ~79 GPU-h). EuroLLM: the alignment start (the GOLDW recipe with the new
  decoder, artemis `336423`) first, then SD24 with `REPEAT_FROM_GOLD=1` and all 24 languages as
  translate targets.
- **Scoring.** `eval/score_sd24_run.sh <run>` pulls a finished run from MN5 and scores it on the
  FLEURS-24 dev sets (X->en, en->X, ASR) under both prompt formats: `eval/format-repeatfirst.yaml`
  (the SD24 / CE24 training instructions) and `eval/format-sft-random.yaml` (the random templates
  SFT-EU24 was trained and scored with).
- **Reference: SFT-EU24 from A, final** (full fine-tuning, FLEURS dev, lower-cased scoring, random
  templates; A at step 0 in brackets). ASR WER on en/de/fr/es/it 0.063 (0.081), WER on the other 19
  languages 0.179 (1.007), CER 0.075 (0.582); X->en chrF 59.8 on de/fr/es/it and 53.2 on the other
  sources (25.7 / 22.3); en->X chrF 54.1 and 40.7 (28.9 / 20.0). In-domain WER (sd-dev-per-source)
  de .113, en .092, es .067, fr .098, it .135 (A: .125 .087 .068 .104 .148).
- **Results for SD24, CE24 and the EuroLLM run:** not in yet.

## Running on artemis

The h200 node (`hades`) does not mount `/mnt/home`. Submit from the clone at
`/mnt/scratch-artemis/giuseppe/melt-proj/training-sdw`, and export
`WANDB_API_KEY` and `HF_TOKEN` in the submitting shell (W&B reads `~/.netrc`,
which the job cannot see). A job submitted from `/mnt/home` fails in 0 s and
leaves no log.

```bash
DRY_RUN=1 bash projects/self-distill/launch_goldw.sh
bash projects/self-distill/launch_goldw.sh
bash projects/self-distill/launch_teacher_audit.sh
SEED=43 ARM=opd WARM_START=/workspace/outputs/<GOLDW run> bash projects/self-distill/launch_step2.sh
FORMAT=bare bash projects/self-distill/eval/launch_eval_step2.sh /mnt/scratch-artemis/giuseppe/melt-data/outputs/<GOLDW run>
```

The data seed must be on the command line: `MELT_SEED`, which `run_train.sh`
turns into a `shard_seed` override, is not forwarded into the container
(`bash/run_train_singularity.sbatch` has no `container_env MELT_SEED`), so a
container run reads the YAML's `seed`/`shard_seed` (42) and logs `MELT_SEED=42`
whatever the host set. `launch_step2.sh` passes `DATA_SEED` (default `SEED`) as
`data.train_ds.seed` and `shard_seed`, and names the run `-s<seed>-d<data seed>`.

A mid-run `checkpoint-N` carries no processor or tokenizer: build them with
`save_run_processor.py`, and give melt-eval `-M processor=`, `-T tokenizer=` and
`-T format_config=` (without `-T tokenizer` it dies before the first sample).

Outputs: GOLDW under `/mnt/scratch-artemis/giuseppe/melt-data/outputs/GOLDW-*`,
the audit under `.../outputs/sd-teacher-audit/<run>/` (`summary.md`,
`summary.json`, `samples.jsonl`).
