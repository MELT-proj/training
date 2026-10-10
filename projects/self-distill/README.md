# Self-distillation, phase 2

Goal: show that on-policy distillation (OPD) from the frozen decoder reading
the transcript improves audio-text alignment beyond gold-transcript CE and
AZeroS-style hard-label distillation (arXiv 2601.06086), for ASR and speech
translation, across languages. The trainer is `melt.training.train_self_distill`
(`melt/training/self_distill.py`, design notes in `docs/self_distillation.md`).

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
3. **Task-conditioned multilingual OPD**: an instruction mix (repeat, translate
   into each language, one generic), targets from the teacher, against the same
   mix trained offline at equal compute.
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
     without the anchor, so a gap to SIFT cannot be the gold CE's.
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
there at 1,000 steps (on-policy: 52.8 / 42.1). Single seed so far; seed-44
replicas of both arms and an offline soft-label arm (`soft_anchor`) are running.

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
