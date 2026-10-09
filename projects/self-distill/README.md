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
2. **OPD vs CE for ASR**, all from one GOLDW checkpoint, equal steps: gold CE
   continued; offline AZeroS (greedy teacher replies with EOS, precomputed); OPD
   with a privileged teacher (transcript + the student's instruction); OPD with
   a small gold-CE anchor. Scored on full-set WER with S/D/I and runaway,
   within-language retrieval, and zero-shot "translate" as a transfer probe.
3. **Task-conditioned multilingual OPD**: an instruction mix (repeat, translate
   into each language, one generic), targets from the teacher, against the same
   mix trained offline at equal compute.

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
```

Outputs: GOLDW under `/mnt/scratch-artemis/giuseppe/melt-data/outputs/GOLDW-*`,
the audit under `.../outputs/sd-teacher-audit/<run>/` (`summary.md`,
`summary.json`, `samples.jsonl`).
