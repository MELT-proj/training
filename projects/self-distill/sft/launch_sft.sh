#!/bin/bash
#
# Self-distillation SFT: supervised fine-tuning of Whisper-large-v3 + MLP adapter (stack_factor 5)
# + Qwen3.5-2B, all three components trainable, from a warm start, on one of the two data
# conditions rendered in this directory. Four runs = two inits x two mixes:
#
#   INIT=goldw   GOLDW-final: the gold-CE ASR warm start (encoder and decoder were frozen there)
#   INIT=sd      the self-distilled warm start; S3 azeros_anchor unless SD_CKPT says otherwise
#   INIT=none    no warm start, no modality alignment: no --model.ckpt, so train.py builds the model the way
#                launch_goldw.sh did before GOLDW's alignment, Whisper-large-v3 and the decoder as the hub ships
#                them under a freshly initialised adapter (seeded by SEED), with the same processor and tokenizer.
#                The run name says "noma"
#   MIX=hr5      SFT-HR5.yaml    en/de/fr/es/it ASR, X->en and en->X for de/fr/es/it
#   MIX=eu24     SFT-EU24.yaml   all 24 EU languages ASR and speech translation, the same total hours
#
# Steps are DERIVED from the config's data.train_ds.total_hours, so a run is one pass over the mix
# whatever world size, BATCH_DURATION and GRAD_ACCUM it is launched with:
#
#   max_steps        = ceil(total_hours * 3600 / (BATCH_DURATION * world_size * GRAD_ACCUM))
#   num_decay_steps  = max_steps // 5     the WSD decay is the last fifth, as in launch_step2.sh
#   save_steps       = max_steps // 16    sixteen checkpoints
#   eval_steps       = max_steps // 32    thirty-two evals (plus the one on start)
#
# save_total_limit is null, so EVERY checkpoint is kept: nothing deletes them, the operator prunes
# (each carries weights and AdamW state for all three components, tens of GB).
#
# Batches fill to less than BATCH_DURATION, so the audio a run really sees is the train_hours counter
# in its log, not max_steps x BATCH_DURATION x world_size x GRAD_ACCUM.
#
# Inputs (environment):
#   INIT, MIX              required, see above
#   STOP_AFTER_STEP        end the run after this optimizer step, a checkpoint saved there (run.stop_after_step).
#                          The LR schedule stays that of MAX_STEPS, so the steps trained are the first steps of the
#                          full run and RESUME=1 can carry it on. Unset (the default): the run goes to MAX_STEPS and
#                          this script prints and passes nothing different
#   SITE                   infra/runners/sites/<site>.sh                         (default mn5)
#   MELT_NODES             nodes                                                 (default 2)
#   MELT_GPUS_PER_NODE     pinned: SLURM_GPUS_ON_NODE lies on MN5's whole-node acc (default 4)
#   BATCH_DURATION         seconds of audio per rank per micro-batch             (default 60)
#   GRAD_ACCUM             micro-batches per optimizer step                      (default 4)
#   MELT_QOS, MELT_TIME    read by the site file. MELT_TIME has no default here: the site's is
#                          one hour, and a real run submitted without one would die at its first
#                          checkpoint. Required unless DRY_RUN=1.
#   SD_CKPT, GOLDW_CKPT    warm-start run roots as the CONTAINER sees them (/workspace/outputs/<run>);
#                          SD_CKPT is how the sd init moves from S3 azeros_anchor to anything else
#   INIT_TAG               the init's word in the run name (default: goldw, or sd-<arm> read off the
#                          S<n>-<arm>[_anchor]-... name of the sd run root, so two sd inits never
#                          share an output directory and silently resume each other)
#   SEED                   trainer.seed, data seed and shard seed                (default 1337)
#   MAX_STEPS, EVAL_STEPS, SAVE_STEPS
#                          override the derived value (a smoke test). num_decay_steps follows
#                          MAX_STEPS, and an overridden MAX_STEPS adds -st<steps> to the run name so a
#                          smoke test can never be resumed by the real run of the same condition.
#   EXP_NAME               replaces the whole run name
#   RESUME=1               continue the run EXP_NAME from the newest checkpoint-* in its output directory
#                          (the sampler state is restored, so the world size must match)
#   DRY_RUN=1              print the plan and the full command, submit nothing
#
# Extra arguments go to the end of the train command, where they win over everything above
# (OmegaConf keeps the last occurrence of a key).
#
# The output directory. train.py refuses to start into a directory that holds anything, and EVERY rank
# makes that check, while the global master writes resolved_config.json into the directory as soon as
# ITS check has passed. On two nodes the ranks of the second one load the warm start later, find the
# master's file and die within two minutes:
#   [rank4]: ValueError: Output directory (...) already exists and is not empty.
# (MN5 acc_debug smoke test, 2 nodes, --model.ckpt.) So a fresh run passes --trainer.overwrite_output_dir
# true, which skips that check, and this script makes it instead, once, on the host before anything is
# queued: an existing directory is an error. RESUME=1 is the way to continue one: it hands train.py the
# directory, which then resumes from the newest checkpoint-* and never makes the check.
#
#   DRY_RUN=1 INIT=goldw MIX=eu24 bash projects/self-distill/sft/launch_sft.sh
#   DRY_RUN=1 INIT=none MIX=eu24 STOP_AFTER_STEP=2872 bash projects/self-distill/sft/launch_sft.sh
#   MELT_TIME=48:00:00 INIT=sd MIX=hr5 bash projects/self-distill/sft/launch_sft.sh
#   RESUME=1 MELT_TIME=48:00:00 INIT=sd MIX=hr5 bash projects/self-distill/sft/launch_sft.sh
#
#   # artemis H200, a pair side by side: the gpu-h200 QOS caps a user at 4 GPUs, so 2 each, and
#   # GRAD_ACCUM 16 keeps MN5's optimizer step (8 GPUs x 4 x 60 s). Checkpoints go to nyx scratch.
#   SITE=artemis MELT_NODES=1 MELT_GPUS_PER_NODE=2 GRAD_ACCUM=16 MELT_PARTITION=h200 \
#     MELT_QOS=gpu-h200 MELT_TIME=48:00:00 OUTPUT_DIR=/mnt/scratch-nyx/giuseppe/melt/outputs-sft \
#     INIT=sd MIX=hr5 bash projects/self-distill/sft/launch_sft.sh
#
#   # the smoke test: one eval and one save inside 40 steps
#   MELT_QOS=acc_debug MELT_TIME=02:00:00 MAX_STEPS=40 EVAL_STEPS=20 SAVE_STEPS=30 \
#     INIT=goldw MIX=eu24 bash projects/self-distill/sft/launch_sft.sh \
#     --trainer.logging_steps 1 --trainer.eval_on_start false
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

INIT="${INIT:?set INIT=goldw|sd|none}"
MIX="${MIX:?set MIX=hr5|eu24}"
SITE="${SITE:-mn5}"
SEED="${SEED:-1337}"
BATCH_DURATION="${BATCH_DURATION:-60}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ENCODER="${ENCODER:-openai/whisper-large-v3}"
DECODER="${DECODER:-Qwen/Qwen3.5-2B}"
export MELT_NODES="${MELT_NODES:-2}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"
# MELT_SEED becomes --data.*.shard_seed in bash/run_train.sh, but the container shim never forwards
# it, so in a container it is 42 whatever is set here and the shard_seed override below is what
# counts: it sits after run_train.sh's, and the last occurrence of a key wins.
export MELT_SEED="$SEED"
# The default entrypoint, set explicitly so a stale MELT_TRAIN_MODULE in the caller's shell cannot
# swap the trainer under an SFT run.
export MELT_TRAIN_MODULE=melt.training.train
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))

if [[ -z "${MELT_TIME:-}" ]]; then
    [[ "${DRY_RUN:-0}" == "1" ]] || die "set MELT_TIME (HH:MM:SS); the site default is one hour"
    echo "[DRY_RUN] WARNING: MELT_TIME is unset: a real submit refuses to run without it" >&2
fi

# The two data conditions.
case "$MIX" in
    hr5)  CONFIG=projects/self-distill/sft/SFT-HR5.yaml ;;
    eu24) CONFIG=projects/self-distill/sft/SFT-EU24.yaml ;;
    *) die "unknown MIX=$MIX (hr5|eu24)" ;;
esac
[[ -f "$CONFIG" ]] || die "$CONFIG not found: run from a checkout that has it"

# The two warm starts, as the container sees them. B is the one that moves.
DEFAULT_GOLDW_RUN=GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g
DEFAULT_SD_RUN=S3-azeros_anchor-from-GOLDW-final-mix05-bd30-lr2e4-3000st-s43-2g
case "$INIT" in
    goldw) CKPT="${GOLDW_CKPT:-/workspace/outputs/${DEFAULT_GOLDW_RUN}}"; DERIVED_TAG=goldw ;;
    sd)
        CKPT="${SD_CKPT:-/workspace/outputs/${DEFAULT_SD_RUN}}"; DERIVED_TAG=sd
        # S3-azeros_anchor-from-... -> sd-azeros ; S2-sift_anchor-from-... -> sd-sift
        if [[ "$(basename "$CKPT")" =~ ^S[0-9]+-([a-z]+)(_anchor)?- ]]; then
            DERIVED_TAG="sd-${BASH_REMATCH[1]}"
        fi ;;
    none) CKPT=""; DERIVED_TAG=noma ;;
    *) die "unknown INIT=$INIT (goldw|sd|none)" ;;
esac
INIT_ARGS=(--model.ckpt "$CKPT")
[[ "$INIT" != none ]] || INIT_ARGS=()
INIT_TAG="${INIT_TAG:-$DERIVED_TAG}"

# The derived schedule. Exact rational arithmetic: total_hours is a rounded decimal and a float
# product could tip the ceiling by one on a boundary.
PLAN="$(python3 - "$CONFIG" "$BATCH_DURATION" "$WORLD_SIZE" "$GRAD_ACCUM" <<'PY'
import math
import sys
from fractions import Fraction

import yaml

config, batch_duration, world_size, grad_accum = sys.argv[1], *(Fraction(a) for a in sys.argv[2:])
with open(config) as f:
    hours = Fraction(str(yaml.safe_load(f)["data"]["train_ds"]["total_hours"]))
if hours <= 0:
    sys.exit(f"{config}: data.train_ds.total_hours is {hours}; render it with build_sft_mix.py")
print(float(hours), math.ceil(hours * 3600 / (batch_duration * world_size * grad_accum)))
PY
)" || die "could not derive max_steps from $CONFIG"
read -r TOTAL_HOURS DERIVED_STEPS <<<"$PLAN"

MAX_STEPS="${MAX_STEPS:-$DERIVED_STEPS}"
DECAY_STEPS=$(( MAX_STEPS / 5 ))
SAVE_STEPS="${SAVE_STEPS:-$(( MAX_STEPS / 16 ))}"
EVAL_STEPS="${EVAL_STEPS:-$(( MAX_STEPS / 32 ))}"
(( SAVE_STEPS >= 1 )) || SAVE_STEPS=1
(( EVAL_STEPS >= 1 )) || EVAL_STEPS=1
STEPS_TAG=""
[[ "$MAX_STEPS" == "$DERIVED_STEPS" ]] || STEPS_TAG="-st${MAX_STEPS}"
STOP_ARGS=()
if [[ -n "${STOP_AFTER_STEP:-}" ]]; then
    [[ "$STOP_AFTER_STEP" =~ ^[1-9][0-9]*$ ]] || die "STOP_AFTER_STEP=$STOP_AFTER_STEP is not a positive integer"
    (( STOP_AFTER_STEP <= MAX_STEPS )) || die "STOP_AFTER_STEP=$STOP_AFTER_STEP is beyond MAX_STEPS=$MAX_STEPS: the run would end first"
    STOP_ARGS=(--run.stop_after_step "$STOP_AFTER_STEP")
fi
EXP_NAME="${EXP_NAME:-SFT-${MIX^^}-from-${INIT_TAG}-bd${BATCH_DURATION}-ga${GRAD_ACCUM}-${WORLD_SIZE}g-s${SEED}${STEPS_TAG}}"

# The weights and model config come from model.ckpt, but train.py still builds a processor from
# model.encoder/decoder.name before loading it, reads the encoder attention backend from the YAML (a
# checkpoint records none), and saves resolved_config.json from these values -- so describe the stack
# exactly as launch_step2.sh does. The tables are plan_arm's: no second copy of them here.
STACK_OUT="$(python3 - "${REPO_ROOT}/projects/ablation-campaign" "$DECODER" "$ENCODER" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from plan_arm import DECODER_PROFILES, ENCODER_ATTN_IMPLEMENTATION, ENCODER_WINDOW_FRAMES

decoder, encoder = sys.argv[2], sys.argv[3]
if decoder not in DECODER_PROFILES:
    sys.exit(f"DECODER={decoder!r} is not in plan_arm.DECODER_PROFILES")
p = DECODER_PROFILES[decoder]
pairs = [("model.decoder.name", decoder), ("model.decoder.eos_token", p["eos_token"]),
         ("model.decoder.pad_token", p["pad_token"]), ("data.chat_template_config", p["chat_template_config"])]
if "chat_template_from" in p:
    pairs.append(("model.decoder.chat_template_from", p["chat_template_from"]))
pairs.append(("model.encoder.name", encoder))
if encoder in ENCODER_WINDOW_FRAMES:
    pairs.append(("model.encoder.max_audio_seq_len", ENCODER_WINDOW_FRAMES[encoder]))
if encoder in ENCODER_ATTN_IMPLEMENTATION:
    pairs.append(("model.encoder.attn_implementation", ENCODER_ATTN_IMPLEMENTATION[encoder]))
for key, value in pairs:
    print(f"--{key}")
    print(value)
PY
)" || die "could not read the stack overrides from plan_arm"
mapfile -t STACK_OVERRIDES <<<"$STACK_OUT"
(( ${#STACK_OVERRIDES[@]} > 0 )) || die "plan_arm produced no stack overrides"

# The warm start must exist where the container will look: the site's OUTPUT_DIR on the host. Checked
# here so a typo costs a second, not a queue wait.
HOST_OUTPUT_DIR="$( . "${REPO_ROOT}/infra/runners/sites/${SITE}.sh" >/dev/null 2>&1 && printf '%s' "${OUTPUT_DIR:-}" )"
case "$CKPT" in
    /workspace/outputs/*) HOST_CKPT="${HOST_OUTPUT_DIR}/${CKPT#/workspace/outputs/}" ;;
    *) HOST_CKPT="" ;;
esac
if [[ -n "$HOST_CKPT" && ! -f "${HOST_CKPT}/model.safetensors" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] WARNING: no ${HOST_CKPT}/model.safetensors on this host" >&2
    else
        die "no ${HOST_CKPT}/model.safetensors: stage the warm start under the site's OUTPUT_DIR first"
    fi
fi

# The run directory (see the header for why this is checked here and not by train.py).
HOST_RUN_DIR=""
[[ -n "$HOST_OUTPUT_DIR" ]] && HOST_RUN_DIR="${HOST_OUTPUT_DIR}/${EXP_NAME}"
if [[ "${RESUME:-0}" == "1" ]]; then
    RUN_DIR_ARGS=(--trainer.resume_from_checkpoint "/workspace/outputs/${EXP_NAME}")
    if [[ -n "$HOST_RUN_DIR" && -z "$(ls -d "${HOST_RUN_DIR}"/checkpoint-* 2>/dev/null)" ]]; then
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "[DRY_RUN] WARNING: RESUME=1 but no ${HOST_RUN_DIR}/checkpoint-* on this host" >&2
        else
            die "RESUME=1 but there is no checkpoint-* in ${HOST_RUN_DIR}: nothing to continue"
        fi
    fi
else
    RUN_DIR_ARGS=(--trainer.overwrite_output_dir true)
    if [[ -n "$HOST_RUN_DIR" && -n "$(ls -A "$HOST_RUN_DIR" 2>/dev/null)" ]]; then
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "[DRY_RUN] WARNING: ${HOST_RUN_DIR} already exists; a real submit refuses (RESUME=1 or another EXP_NAME)" >&2
        else
            die "${HOST_RUN_DIR} already exists: RESUME=1 continues that run, another EXP_NAME starts a new one"
        fi
    fi
fi

# --model.master_dtype float32: both warm starts load as bf16 (transformers 5 loads model.ckpt with
# dtype="auto"), and DDP does not upcast, so without it the run trains in pure bf16, where AdamW
# updates at the encoder's 6e-6 and the decoder's 2e-5 mostly round to nothing (a 40-step smoke
# test wrote optimizer state at 4 bytes per parameter). Autocast keeps the compute in bf16.
# --trainer.ddp_gradient_as_bucket_view true: fp32 masters peak at 62.7 GiB per GPU at 60 s on
# H200 (the HR5 pair, launched without it); sharing the gradients with DDP's buckets saves ~9.4
# GiB, which is what lets the same recipe fit a 64 GB MN5 H100. The arithmetic is unchanged.
CMD=(
    infra/runners/submit-container.sh "$SITE" config/accelerate/ddp.yaml
    --config "$CONFIG"
    --run.exp_name "$EXP_NAME"
    --trainer.output_dir "/workspace/outputs/${EXP_NAME}"
    "${RUN_DIR_ARGS[@]}"
    ${INIT_ARGS[@]+"${INIT_ARGS[@]}"}
    --model.master_dtype float32
    "${STACK_OVERRIDES[@]}"
    --model.adapter.stack_factor 5
    --data.train_ds.batch_duration "$BATCH_DURATION"
    --trainer.gradient_accumulation_steps "$GRAD_ACCUM"
    --trainer.gradient_checkpointing true
    --trainer.ddp_gradient_as_bucket_view true
    --trainer.max_steps "$MAX_STEPS"
    --trainer.lr_scheduler_kwargs.num_decay_steps "$DECAY_STEPS"
    --trainer.eval_steps "$EVAL_STEPS"
    --trainer.save_steps "$SAVE_STEPS"
    --trainer.save_total_limit null
    --trainer.seed "$SEED"
    --data.train_ds.seed "$SEED"
    --data.train_ds.shard_seed "$SEED"
    ${STOP_ARGS[@]+"${STOP_ARGS[@]}"}
    "$@"
)

echo "[launch_sft] ${EXP_NAME}"
if [[ "$INIT" == none ]]; then
    echo "[launch_sft] init:  none (the hub's Whisper and decoder weights, a fresh adapter seeded ${SEED})"
else
    echo "[launch_sft] init:  ${CKPT}"
fi
echo "[launch_sft] mix:   ${CONFIG} (${TOTAL_HOURS} h)"
echo "[launch_sft] world: ${MELT_NODES} nodes x ${MELT_GPUS_PER_NODE} GPUs = ${WORLD_SIZE}; ${BATCH_DURATION} s x ${GRAD_ACCUM} accum = $(( BATCH_DURATION * GRAD_ACCUM * WORLD_SIZE )) s of audio per optimizer step"
echo "[launch_sft] steps: max ${MAX_STEPS} (decay ${DECAY_STEPS}), eval every ${EVAL_STEPS}, save every ${SAVE_STEPS}, every checkpoint kept; seed ${SEED}"
if [[ "${RESUME:-0}" == "1" ]]; then
    echo "[launch_sft] mode:  RESUME from the newest checkpoint-* in /workspace/outputs/${EXP_NAME}"
else
    echo "[launch_sft] mode:  fresh run into /workspace/outputs/${EXP_NAME} (absent or empty on this host)"
fi
(( ${#STOP_ARGS[@]} == 0 )) || echo "[launch_sft] stop:  after step ${STOP_AFTER_STEP}, a checkpoint saved there; the LR schedule stays that of the ${MAX_STEPS} steps"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] MELT_TRAIN_MODULE=${MELT_TRAIN_MODULE} MELT_NODES=${MELT_NODES} MELT_GPUS_PER_NODE=${MELT_GPUS_PER_NODE} MELT_QOS=${MELT_QOS:-<site default>} MELT_TIME=${MELT_TIME:-<unset>}"
    printf '[DRY_RUN] would run:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

exec "${CMD[@]}"
