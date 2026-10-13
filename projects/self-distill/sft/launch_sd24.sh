#!/bin/bash
#
# Adapter-only self-distillation over the 24 EU languages ("SD24"): the offline, instruction-mixed
# recipe of launch_step2.sh (half "repeat", half "translate into X", the frozen decoder's greedy reply
# as the target) on the ASR half of EU24, for far more hours than the 50 h of the five-language runs.
# Only the adapter trains. The question it answers: can this replace the SFT step? Its scored
# counterpart is SFT-EU24 (all three components, gold ASR + ST labels, from the same GOLDW-final).
#
#   ARM=azeros_anchor   offline CE on the teacher's greedy reply + 0.5 x gold CE on the repeat prompt (default)
#   ARM=azeros          the same without the gold anchor
#   ARM=ce              the matched control: the adapter alone on the gold transcript under the repeat
#                       instruction (the standard MA loss, no mix) over the same data in the same order
#
# The data is projects/self-distill/sft/SD-EU24-<HOURS>h.yaml (build_sft_mix.py --sd24-hours <HOURS>).
# Steps are DERIVED from its data.train_ds.total_hours, as in launch_sft.sh, so a run is one pass over
# the mix whatever the world size and BATCH_DURATION:
#
#   max_steps = ceil(total_hours * 3600 / (BATCH_DURATION * world_size))      (no gradient accumulation)
#   num_decay_steps = max_steps // 5, save every max_steps // 8, eval every max_steps // 16
#
# Batches fill to less than BATCH_DURATION, so the audio a run really sees is the train_hours counter
# in its log, a little under HOURS.
#
# Inputs (environment):
#   HOURS              total hours of the rendered mix                       (default 2700)
#   ARM, SITE          see above, and infra/runners/sites/<site>.sh          (defaults azeros_anchor, mn5)
#   MELT_NODES, MELT_GPUS_PER_NODE   (defaults 2 and 4; pinned: SLURM_GPUS_ON_NODE lies on MN5's acc)
#   BATCH_DURATION     seconds of audio per rank per step                    (default 120)
#   ADAPTER_LR         peak adapter LR, WSD schedule as launch_step2.sh      (default 2e-4)
#   TRANSLATE_FRAC     share of utterances given a translate instruction     (default 0.5; 0 for ARM=ce)
#   TRANSLATE_TARGETS  comma-separated ISO codes a translation may target    (default: all 24 EU languages)
#   MASTER_DTYPE       model.master_dtype (default float32; "none" skips it). The warm start loads as pure
#                      bf16 under DDP, where AdamW updates below half a bf16 ulp are lost -- and the WSD decay
#                      takes the LR down to 0.1x. Autocast keeps the compute in bf16.
#   WARM_START         warm-start run root as the CONTAINER sees it          (default: GOLDW-final, "A")
#   MELT_QOS, MELT_TIME  read by the site file. MELT_TIME has no default here (the site's is one hour); it
#                      is required unless DRY_RUN=1.
#   SEED               trainer seed, data seed and shard seed                (default 1337)
#   MAX_STEPS, EVAL_STEPS, SAVE_STEPS   override the derived value (a smoke test). An overridden MAX_STEPS
#                      adds -st<steps> to the run name, so a smoke test can never be resumed by the real run.
#   EVAL_SAMPLES       clips per validation language                         (default 20)
#   EXP_NAME           replaces the whole run name
#   RESUME=1           continue EXP_NAME from the newest checkpoint-* in its output directory
#   DRY_RUN=1          print the plan and the full command, submit nothing
#
# Extra arguments go to the end of the train command, where they win over everything above.
#
# The output directory: train.py makes an "exists and is not empty" check on EVERY rank while the master
# writes resolved_config.json as soon as ITS check passed, so on two nodes a fresh run needs
# --trainer.overwrite_output_dir true. This script makes the check once, on the host, before anything is
# queued (the same arrangement as launch_sft.sh).
#
#   DRY_RUN=1 bash projects/self-distill/sft/launch_sd24.sh
#   # the smoke test: one eval and one save inside 40 steps
#   MELT_QOS=acc_debug MELT_TIME=02:00:00 MAX_STEPS=40 EVAL_STEPS=20 SAVE_STEPS=30 \
#     bash projects/self-distill/sft/launch_sd24.sh --trainer.logging_steps 1
#   MELT_TIME=48:00:00 bash projects/self-distill/sft/launch_sd24.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

ARM="${ARM:-azeros_anchor}"
HOURS="${HOURS:-2700}"
SITE="${SITE:-mn5}"
SEED="${SEED:-1337}"
BATCH_DURATION="${BATCH_DURATION:-120}"
ADAPTER_LR="${ADAPTER_LR:-2e-4}"
TRANSLATE_FRAC="${TRANSLATE_FRAC:-0.5}"
ALL_EU24="bg,cs,da,de,el,en,es,et,fi,fr,ga,hr,hu,it,lt,lv,mt,nl,pl,pt,ro,sk,sl,sv"
TRANSLATE_TARGETS="${TRANSLATE_TARGETS:-$ALL_EU24}"
MASTER_DTYPE="${MASTER_DTYPE:-float32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20}"
export MELT_NODES="${MELT_NODES:-2}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))

case "$ARM" in
    azeros|azeros_anchor) ;;
    ce) TRANSLATE_FRAC=0 ;;
    *) die "unknown ARM=$ARM (azeros|azeros_anchor|ce)" ;;
esac
if [[ -z "${MELT_TIME:-}" ]]; then
    [[ "${DRY_RUN:-0}" == "1" ]] || die "set MELT_TIME (HH:MM:SS); the site default is one hour"
    echo "[DRY_RUN] WARNING: MELT_TIME is unset: a real submit refuses to run without it" >&2
fi

CONFIG="projects/self-distill/sft/SD-EU24-${HOURS}h.yaml"
[[ -f "$CONFIG" ]] || die "$CONFIG not found: render it with build_sft_mix.py --sd24-hours ${HOURS}"

DEFAULT_GOLDW_RUN=GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g
WARM_START="${WARM_START:-/workspace/outputs/${DEFAULT_GOLDW_RUN}}"

# The derived schedule, in exact rational arithmetic (total_hours is a rounded decimal).
PLAN="$(python3 - "$CONFIG" "$BATCH_DURATION" "$WORLD_SIZE" <<'PY'
import math
import sys
from fractions import Fraction

import yaml

config, batch_duration, world_size = sys.argv[1], *(Fraction(a) for a in sys.argv[2:])
with open(config) as f:
    hours = Fraction(str(yaml.safe_load(f)["data"]["train_ds"]["total_hours"]))
if hours <= 0:
    sys.exit(f"{config}: data.train_ds.total_hours is {hours}; render it with build_sft_mix.py")
print(float(hours), math.ceil(hours * 3600 / (batch_duration * world_size)))
PY
)" || die "could not derive max_steps from $CONFIG"
read -r TOTAL_HOURS DERIVED_STEPS <<<"$PLAN"

export MAX_STEPS="${MAX_STEPS:-$DERIVED_STEPS}"
export SAVE_STEPS="${SAVE_STEPS:-$(( MAX_STEPS / 8 ))}"
export EVAL_STEPS="${EVAL_STEPS:-$(( MAX_STEPS / 16 ))}"
(( SAVE_STEPS >= 1 )) || SAVE_STEPS=1
(( EVAL_STEPS >= 1 )) || EVAL_STEPS=1
STEPS_TAG=""
[[ "$MAX_STEPS" == "$DERIVED_STEPS" ]] || STEPS_TAG="-st${MAX_STEPS}"
MIX_TAG=""
awk "BEGIN { exit !(${TRANSLATE_FRAC} > 0) }" && MIX_TAG="-mix$(echo "$TRANSLATE_FRAC" | tr -d '.')"
export EXP_NAME="${EXP_NAME:-SD24-${ARM}-from-GOLDW-final${MIX_TAG}-${HOURS}h-bd${BATCH_DURATION}-${WORLD_SIZE}g-s${SEED}${STEPS_TAG}}"

# The warm start and the run directory must be where the container will look: the site's OUTPUT_DIR on
# the host. Checked here so a typo costs a second, not a queue wait.
HOST_OUTPUT_DIR="$( . "${REPO_ROOT}/infra/runners/sites/${SITE}.sh" >/dev/null 2>&1 && printf '%s' "${OUTPUT_DIR:-}" )"
case "$WARM_START" in
    /workspace/outputs/*) HOST_CKPT="${HOST_OUTPUT_DIR}/${WARM_START#/workspace/outputs/}" ;;
    *) HOST_CKPT="" ;;
esac
if [[ -n "$HOST_CKPT" && ! -f "${HOST_CKPT}/model.safetensors" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] WARNING: no ${HOST_CKPT}/model.safetensors on this host" >&2
    else
        die "no ${HOST_CKPT}/model.safetensors: stage the warm start under the site's OUTPUT_DIR first"
    fi
fi
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

EXTRAS=("${RUN_DIR_ARGS[@]}")
[[ "$MASTER_DTYPE" == none ]] || EXTRAS+=(--model.master_dtype "$MASTER_DTYPE")
# The value is quoted for YAML, as launch_step2.sh quotes the templates: a bare list of ISO codes is one
# string to OmegaConf, but a single code such as "no" would become a boolean.
awk "BEGIN { exit !(${TRANSLATE_FRAC} > 0) }" && EXTRAS+=(--distill.translate_targets "\"${TRANSLATE_TARGETS}\"")

echo "[launch_sd24] ${EXP_NAME}"
echo "[launch_sd24] arm:   ${ARM}, translate_frac ${TRANSLATE_FRAC}, targets ${TRANSLATE_TARGETS}"
echo "[launch_sd24] init:  ${WARM_START}"
echo "[launch_sd24] mix:   ${CONFIG} (${TOTAL_HOURS} h)"
echo "[launch_sd24] world: ${MELT_NODES} nodes x ${MELT_GPUS_PER_NODE} GPUs = ${WORLD_SIZE}; ${BATCH_DURATION} s per rank = $(( BATCH_DURATION * WORLD_SIZE )) s of audio per step"
echo "[launch_sd24] steps: max ${MAX_STEPS} (decay $(( MAX_STEPS / 5 ))), eval every ${EVAL_STEPS}, save every ${SAVE_STEPS}; adapter LR ${ADAPTER_LR}; seed ${SEED}; master dtype ${MASTER_DTYPE}"

export ARM TRANSLATE_FRAC SITE SEED BATCH_DURATION ADAPTER_LR EVAL_SAMPLES WARM_START
export CONFIG
exec bash projects/self-distill/launch_step2.sh "${EXTRAS[@]}" "$@"
