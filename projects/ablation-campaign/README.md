# Ablation campaign

Cross-language ablation of the MA (modality alignment) -> IFT (instruction
fine-tuning) pipeline. This file documents the "campaign grouping" launcher
layout (data / architecture / optimisation axes, one launcher per stage) --
for the data-mixture design itself, read `build_campaign_config.py`'s module
docstring.

## The three axes

The campaign varies three things. Each lives in exactly one place:

| axis | varies how | belongs in |
|---|---|---|
| **Data**: mixture, hours, task | a rendered config, ~550 lines | one YAML per budget x task: `ABL-MA-125-asr.yaml`, `ABL-MA-700-asr.yaml`, `ABL-IFT-125.yaml` |
| **Architecture**: adapter, encoder, decoder, freezing | 0-7 CLI overrides | `ADAPTER`, `ADAPTER_FREEZE`, `ENCODER`, `ENCODER_FREEZE`, `DECODER`, `DECODER_FREEZE` env vars |
| **Optimisation**: adapter LR | 0-1 CLI override | `ADAPTER_LR` env var |

There is deliberately **no YAML per arm**. A data axis change (a new budget or
task mix) is big enough, and shared enough across many arms, to earn its own
rendered config via `build_campaign_config.py`. Architecture and LR changes
are a handful of keys and belong on the command line, where they show up in
`resolved_config.json` and in `arms.tsv` (below) without anyone having to diff
two 550-line YAMLs to find them.

## Launching an arm

```bash
# stage 1 (MA), 125 h/language, default architecture (whatever
# ABL-MA-125-asr.yaml itself declares)
CONFIG=projects/ablation-campaign/ABL-MA-125-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh

# stage 1 (MA), 700 h/language, ablating the adapter LR
CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml ADAPTER_LR=1e-4 \
    bash projects/ablation-campaign/launch_MA.sh

# stage 2 (IFT)
CONFIG=projects/ablation-campaign/ABL-IFT-125.yaml \
    bash projects/ablation-campaign/launch_IFT.sh

# see the exact command without submitting anything
DRY_RUN=1 CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml \
    bash projects/ablation-campaign/launch_MA.sh
```

`launch_MA.sh` and `launch_IFT.sh` are thin wrappers (set `STAGE` and a
per-stage default `CONFIG`) around the shared `launch_campaign.sh`, which does
the real work: compose `EXP_NAME`, derive `eval_steps`/`save_steps`, work out
which architecture overrides are actually needed, and call
`infra/runners/submit-container.sh`. MA and IFT differ only in which module is
frozen by convention (see below) and in their default `CONFIG` -- everything
else is identical, so one core script with a `STAGE` variable seemed better
than duplicating the whole launch/registry/dry-run mechanics twice.

**Every architecture/optimisation env var defaults to empty, meaning
"inherit whatever the chosen `CONFIG` already says."** No CLI override is
emitted, and `EXP_NAME` reflects the real, effective value either way. Set one
explicitly to run an ablation; it turns into a CLI override only if the
requested value actually differs from the config (which is also why the
acceptance-test parity check in the PR description works -- the two old,
hand-written launchers had no architecture overrides beyond the 125 h arm's
decoder-and-chat-template bundle, and this launcher reproduces exactly that
when asked for the same values).

This was a deliberate choice over a single "sensible default" set of freeze
flags: MA trains only the adapter (encoder and decoder frozen), while IFT
freezes the adapter stage 1 produced and trains the decoder on top of it (see
`ABL-IFT-125.yaml`'s own `model.adapter`/`model.decoder` comments). A
hardcoded MA-shaped default (encoder/decoder frozen, adapter trainable) would
have been right for stage 1 and silently backwards for stage 2 the first time
someone ran `launch_IFT.sh` without overriding it.

Topology (`MELT_NODES`, `MELT_GPUS_PER_NODE`) and `SEED` are also
overridable; defaults are 2 nodes x 4 GPUs and seed 42, matching every arm run
so far. `MELT_GPUS_PER_NODE` must stay pinned, not autodetected -- MN5
allocates `acc` nodes whole, so `SLURM_GPUS_ON_NODE` reports the node's full
GPU count regardless of what was requested (PR #90).

Any argument after the env vars is passed through as an extra override, e.g.:

```bash
CONFIG=... bash projects/ablation-campaign/launch_MA.sh --trainer.max_grad_norm 2.0
```

## Naming convention

`EXP_NAME` is composed, never typed by hand, from:

```
{STAGE}-{data tag}-{encoder}{F|T}-{decoder}{F|T}-{adapter}{F|T}-{lr tag}-s{seed}-{world_size}g
```

e.g. `MA-125asr-w2vbF-llama1bInsF-mlpT-lr2e4-s42-8g`. Trailing `F`/`T` marks a
module frozen/trainable. The **LR tag is new** in this refactor (adapter LR
is now an ablated axis, so it has to be visible in the name or two arms that
only differ in LR become indistinguishable in W&B and in `outputs/`); every
other segment matches the convention documented in the pre-refactor
`launch_MA_llama32-1b-instruct_mn5.sh` header. The data tag comes straight
from the config's own filename (`ABL-{STAGE}-<tag>.yaml` -> `<tag>` with
dashes stripped), not from parsing hours out of the YAML, since the filename
is already the source of truth for "which budget x task is this."

## Two rules that matter more than the file layout

**Anything that must be identical across arms belongs in the base config,
never the launcher.** `batch_duration`, `quadratic_duration`, `max_samples`
and similar knobs govern effective batch size and eval noise; if one of them
ends up as a launcher default instead of a YAML value, it is only a matter of
time before two arms drift apart on it without anyone deciding that on
purpose. `launch_campaign.sh` only ever overrides the three declared axes plus
the campaign-wide invariants (`num_train_epochs: 1`, `save_total_limit: 2`)
that apply to every arm identically by convention, not per-arm.

**Ablating data breaks step-comparability.** With `num_train_epochs: 1`, a
125 h arm and a 700 h arm derive different step counts by design (903 vs
2188) -- one epoch means different things at different budgets. Compare data
arms at equal **audio hours** (or wall-clock, or `train_hours/*` from W&B),
never at equal step. Architecture and LR arms, by contrast, share a `CONFIG`
and therefore a step count, so they stay directly comparable step-for-step.

## Derived eval_steps / save_steps

`eval_steps` and `save_steps` used to be hand-maintained per arm (100/200 for
the 125 h arm's 903 steps, 200/200 for the 700 h arm's 2188) and drifted
whenever `batch_duration` changed. They are now derived from the config's own
`total_hours`, `batch_duration`, `quadratic_duration` and
`trainer.gradient_accumulation_steps` at the actual world_size, targeting
~11 eval rounds per run:

```
steps = ceil(total_hours * 3600 / batch_duration * inflation / world_size / gradient_accumulation_steps)
eval_steps = save_steps = round(steps / 11)
```

`inflation` corrects for `quadratic_duration`: lhotse charges each cut
`d + d^2/quadratic_duration`, so a batch holds less real audio than
`batch_duration` seconds and `steps` would otherwise be understated (586 vs
the true 903 for `ABL-MA-125-asr.yaml`). `plan_arm.py`'s
`effective_duration_inflation` is a deliberate line-for-line duplicate of
`melt/training/data/audio/lhotse/dataloader.py`'s
`_effective_duration_inflation` -- duplicated rather than imported so this can
run in any dev shell with PyYAML, not just inside the training container
(that module pulls in torch/omegaconf). Verified against the two known-good
step counts: 2188 for `ABL-MA-700-asr.yaml` and 903 for `ABL-MA-125-asr.yaml`.

Note `save_steps == eval_steps` is a simplification versus the old arms (the
125 h launcher used `save_steps = 2 x eval_steps`); see the PR description for
the full before/after diff.

## Arms registry

Every real submission (not `DRY_RUN=1`) appends one line to `arms.tsv`:
UTC timestamp, `EXP_NAME`, SLURM job id, and the exact effective command. The
campaign's deliverable is a comparison table; reconstructing "what exactly did
arm X run" from shell history is how campaigns go wrong. `arms.tsv` is
checked into git as the source of truth -- do not hand-edit it.

## IFT and `max_steps`

`ABL-IFT-125.yaml` currently pins `trainer.max_steps: 6250` in the base
config, computed (per its own comment) at world_size 2. `launch_IFT.sh` does
NOT override `max_steps` -- see that script's header for why this is flagged
rather than silently fixed.
