# Online self-distillation for modality alignment

`melt.training.train_self_distill` trains the adapter of a frozen
encoder / frozen decoder speech LLM on the decoder's own responses instead of
on gold transcripts. It is the online counterpart of AZeroS's SIFT
([arXiv 2601.06086](https://arxiv.org/abs/2601.06086)), and the standard MA
path (`melt.training.train`) is untouched by it.

## The method in one paragraph

AZeroS feeds the frozen LLM the transcript as text, with no instruction,
stores its response `y`, and trains the projector with cross-entropy on `y`
given the audio (their Eq. 3). Here the response is produced during training
instead: per micro-batch, with probability `distill.lmbda` it is sampled from
the **student** (audio path, instruction-free `{audio_token}` prompt;
on-policy), otherwise from the **teacher** (the same decoder on the text path,
prompted with the transcript). Both paths then score the same response tokens,
and the loss is TRL's generalized JSD between the two full next-token
distributions (`GKDTrainer.generalized_jsd_loss`; `beta=0` forward KL,
`beta=1` reverse KL), or hard-label cross-entropy (`loss: ce`, teacher
rollouts only), which is exactly AZeroS's objective.

## Why the teacher costs nothing

The teacher is `model.text_decoder` called on token ids: no adapter, no audio,
no second copy of the weights. That holds only while the decoder is frozen and
has no LoRA (otherwise the target moves with the student), so the trainer
refuses any trainable parameter outside the adapter, and checks after the
first backward that only adapter parameters received a gradient. Logits are
sliced to the decoder's original vocabulary (MELT's added audio tokens and the
padding rows of the resized, tied head are excluded, and suppressed in
generation), which makes the teacher exactly the original LLM.

## Why not GRPO

GRPO needs a scalar reward per sampled sequence and a group of samples per
prompt. Here the teacher's full distribution at every response position is
available for one extra forward pass, because student and teacher share a
tokenizer and a decoder. A per-token KL uses all of it, with no group sampling
and no advantage variance. GRPO would reduce that to one number per sequence
(e.g. the teacher log-likelihood of the sample, which is dominated by length)
and pay a G-times generation cost for it. TRL's own on-policy distillation
trainers (GKD, and in TRL 1.x `SDFTTrainer`, the same "one model, privileged
teacher context" setup) take the per-token route. None of them accepts audio
inputs or a Lhotse dataloader, so the trainer here subclasses `MELTTrainer`
and takes only the divergence from TRL.

## Running it

```bash
# print the command, submit nothing
DRY_RUN=1 bash projects/ablation-campaign/launch_self_distill.sh

# online AZeroS (teacher responses, hard labels): the control
LMBDA=0 LOSS=ce bash projects/ablation-campaign/launch_self_distill.sh
```

`projects/ablation-campaign/self-distill.yaml` is an overlay on
`ABL-MA-700-asr.yaml`. It reuses the same data, filters, frozen backbones,
10 Hz MLP adapter and LR schedule as `MA-700-screen-w2vb-lr2e3-b300-evalfrozen`,
so the two arms differ only in the objective. The launcher sets
`MELT_TRAIN_MODULE=melt.training.train_self_distill`, which
`bash/run_train.sh` uses in place of its default module. The image must carry
TRL (the `distill` extra, pinned to 0.29.x because every TRL 1.x needs a newer
`datasets` than the image has).

## What is logged

- `eval_<set>_loss` is replaced by the **alignment gap**: the per-token
  KL(teacher || student) on the teacher's greedy response. It is defined for
  any MELT checkpoint, including gold-transcript MA arms.
- `eval_<set>_wer` uses `validation_ds.prompt_template`, which the overlay sets
  to a transcription instruction. An instruction-free model answers an
  utterance rather than transcribing it, so instruction-free WER means little
  for this method.
- `distill/*` on training rows: on-policy fraction, response length, truncated
  fraction, and the teacher's and student's mean log-probability of the
  response. `distill/teacher_logp` rising while `distill/response_len` falls is
  the signature of the student collapsing to generic text the teacher accepts
  regardless of the transcript.

## Known limits

- Every step generates up to `distill.max_new_tokens` tokens with HF
  `generate` (MELT's audio-embedding model cannot be served by vLLM), so a step
  costs several times an MA step. Measure it before sizing a run.
- Transcripts reach the teacher lowercased, because the dataset lowercases all
  text. AZeroS feeds the raw transcript.
- The method assumes an instruct decoder whose instruction-free reply actually
  restates the content of the input (AZeroS's "self-elicit" condition, their
  §5.7). Check that on the target decoder before training: a decoder that
  answers tersely makes information-sparse targets.
- DDP only. Rollouts and the teacher forward call submodules directly.
