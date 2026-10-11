# Self-distillation, phase 2

Goal: show that on-policy distillation (OPD) from the frozen decoder reading
the transcript improves audio-text alignment beyond gold-transcript CE and
AZeroS-style hard-label distillation (arXiv 2601.06086), for ASR and speech
translation, across languages. The trainer is `melt.training.train_self_distill`
(`melt/training/self_distill.py`, design notes in `docs/self_distillation.md`).

## Where it stands (2026-10-11)

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
  1-2.5 chrF ahead into de and nl, on two data seeds. AZeroS paraphrases instead
  of transcribing, though: in-domain WER 0.160 against 0.11. Our mix keeps
  transcription intact through its repeat half, not the gold-CE term, since
  removing that term changes nothing. So far this is a trade-off, not a
  strict win. AZeroS never evaluated speech translation, so the transfer itself
  is new.
- **Open.** `sift_anchor` (AZeroS plus the gold transcript under the repeat
  instruction) decides whether the translate instructions matter or
  "AZeroS plus transcription training" suffices. Also open: the last seed
  replicas, test sets, spoken QA, and the encoder question (Whisper aligned
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
| text-only ceiling | - | - | 61.4 / 56.6 | - |

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
  (mostly under 0.5) and in-domain WER by 0.003. The first `-s44` runs (same data as `-s43`) agree with
  `-s43` within 0.1-0.4 chrF, which is numerical and sampling noise only.
- **Pending:** `sift_anchor` (d42), `S3-opd_anchor` d44/d45,
  `S3-azeros_anchor` d45 and `S2-sift` d45.

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
