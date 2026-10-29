#!/bin/bash
#
# Adapter-only self-distillation over the 24 EU languages ("SD24"): the offline, instruction-mixed
# recipe of launch_step2.sh (half "repeat", half "translate into X", the frozen decoder's greedy reply
# as the target) on the ASR half of EU24, for far more hours than the 50 h of the five-language runs.
# Only the adapter trains. The question it answers: can this replace the SFT step? Its scored
# counterpart is SFT-EU24 (all three components, gold ASR + ST labels, from the same GOLDW-final).
#
#   ARM=azeros          offline CE on the frozen decoder's greedy reply to the repeat or translate instruction (default).
#                       No gold label enters: the repeat half is gold in effect, because the teacher copies the
#                       transcript exactly in 96-100% of FLEURS clips in all 24 languages (text_ceiling.py --task asr)
#   ARM=azeros_anchor   the same + 0.5 x gold CE on the repeat prompt. The anchor runs a second forward and backward
#                       over the batch and keeps both graphs alive: at BATCH_DURATION 120 it needed more than a 64 GB
#                       H100 has (OOM in _gold_cross_entropy, 61.9 GB allocated; MN5 smoke test 47259527), so give it
#                       a smaller batch
#   ARM=ce              the matched control: the adapter alone on the gold transcript under the repeat
#                       instruction (the standard MA loss, no mix) over the same data in the same order
#   ARM=sift            "SD-SIFT24": AZeroS's own recipe (launch_step2.sh ARM=sift) on the same data: no instruction on
#                       either path. The teacher is the frozen decoder replying freely to the bare transcript, the
#                       student hears {audio_token} alone, no mix, no anchor. The reply is a discussion, not a copy
#                       of the content, so the content-scaled budget and the drop of cut replies (below) are off by
#                       default: a cut reply is the usual case and stays in the loss. Generation-bound: every
#                       micro-batch decodes to MAX_NEW_TOKENS (step 4: 12 s/step on an H200 at 448 tokens)
#
# The data is projects/self-distill/sft/SD-EU24-<HOURS>h.yaml (build_sft_mix.py --sd24-hours <HOURS>).
# Steps are DERIVED from its data.train_ds.total_hours, as in launch_sft.sh, so a run is one pass over
# the mix whatever the world size and BATCH_DURATION:
#
#   max_steps = ceil(total_hours * 3600 / (BATCH_DURATION * world_size * FILL))   (no gradient accumulation)
#   num_decay_steps = max_steps // 5, save every max_steps // 8, eval every max_steps // 16
#
# FILL is the share of BATCH_DURATION a batch really carries: MAX_ROWS caps the cuts of a batch of short
# utterances, so it is below 1. The MN5 smoke test 47263566 measured 0.80 (6.86 h of audio in 60 steps of
# 8 x 64 s). The audio a run really sees is the train_hours counter in its log.
#
# Inputs (environment):
#   HOURS              total hours of the rendered mix                       (default 2700)
#   ARM, SITE          see above, and infra/runners/sites/<site>.sh          (defaults azeros, mn5)
#   MELT_NODES, MELT_GPUS_PER_NODE   (defaults 2 and 4; pinned: SLURM_GPUS_ON_NODE lies on MN5's acc)
#   BATCH_DURATION     seconds of audio per rank per step                    (default 64). 120 and 80 ran out of memory
#                      on a 64 GB H100 even without the anchor (MN5 smoke tests 47260870, 47262184: 61-65 GB peaks on the
#                      worst ranks, 33-42 GB on the others). What is large is the student's logits, padded to
#                      rows x longest reply x a 248k vocabulary and upcast to fp32 by accelerate: 4.76 GiB for 20 rows
#                      of 257 tokens, which one looping reply in a batch of short utterances produces
#   MAX_ROWS           most cuts in one micro-batch (data.train_ds.batch_size) (default 16): bounds that product
#   FILL               share of BATCH_DURATION a batch carries, for the derived steps (default 0.8, see above)
#   REPEAT_FROM_GOLD   1: the repeat rows take the gold transcript as their reply and only the translate rows the teacher's
#                      (distill.repeat_from_gold; also switches the alignment-gap evaluation off). For a teacher that does not
#                      copy its input: EuroLLM-1.7B-Instruct(-2512) answers the repeat instruction with a paraphrase, an
#                      explanation or a refusal for 57-80% of FLEURS clips, Qwen3.5-2B copies 99.5%   (default 0)
#   REPLY_BUDGET_FACTOR  the teacher's reply budget per micro-batch is min(MAX_NEW_TOKENS, factor x the longest content
#                      in tokens + 32)                                        (default 3.5; 0 = the fixed MAX_NEW_TOKENS).
#                      A reply is about as long as its content, up to ~3x for a translation into a finely split language
#   ROLLOUT_LOOKAHEAD  decode the teacher's replies for this many consecutive micro-batches in one call
#                      (distill.rollout_lookahead; default 1 = each micro-batch alone). Greedy decoding is bound by the
#                      latency of a token, not by the number of rows, so for ARM=sift, whose replies run to the cap in
#                      every micro-batch, 4 should cost about a quarter of the decoding time. Needs the fixed budget
#                      (REPLY_BUDGET_FACTOR=0) and no REPEAT_FROM_GOLD; a resume skips up to LOOKAHEAD-1 micro-batches per rank
#   MAX_DURATION       the training loader's longest cut, seconds            (default 30, Whisper's window). The top
#                      10% of the mix is cuts over ~28 s, up to 60 s; their replies hit the token cap, and one such
#                      row makes every rank's rollout run to the cap (the step waits for the slowest rank)
#   MAX_NEW_TOKENS     the teacher's reply budget                            (default 256; 448 cost ~16 s/step at bd120)
#   DROP_TRUNCATED     1: replies cut at MAX_NEW_TOKENS stay out of the loss (distill.drop_truncated); they are
#                      a transcript or translation longer than the budget, or a degenerate loop     (default 1)
#   ADAPTER_LR         peak adapter LR, WSD schedule as launch_step2.sh      (default 2e-4)
#   TRANSLATE_FRAC     share of utterances given a translate instruction     (default 0.25; 0 for ARM=ce). Only the
#                      repeat rows are transcription rows, so this trades ASR audio for translation audio; 0.5 was
#                      the value of every five-language run, which got chrF 53 from ~6 h of X->en rows
#   TRANSLATE_TARGETS  comma-separated ISO codes a translation may target    (default: the 22 EU languages but ga and mt:
#                      the teacher translates into them at chrF 19 / 27 with 25% / 19% runaway replies on FLEURS, and
#                      SFT-EU24 has no ga or mt translation data either; they stay sources and zero-shot targets)
#   MASTER_DTYPE, MASTER_DTYPE_SCOPE   model.master_dtype (default float32; "none" skips it) and which parameters it
#                      covers (default trainable: the adapter alone). The warm start loads as pure bf16 under DDP, where
#                      AdamW updates below half a bf16 ulp are lost -- and the WSD decay takes the LR down to 0.1x.
#                      The frozen encoder and decoder stay bf16, as in every earlier run: 7 GB less memory, no cast of
#                      their weights on every decoding step, and the very teacher B was distilled from. Autocast keeps
#                      the compute in bf16.
#   WARM_START, INIT_TAG   warm-start run root as the CONTAINER sees it, and its word in the run name
#                      (default: GOLDW-final, "A", the Qwen3.5-2B alignment start)
#   DECODER, ENCODER   the stack the warm start was made with (default Qwen/Qwen3.5-2B + openai/whisper-large-v3);
#                      launch_step2.sh reads them, and plan_arm.DECODER_PROFILES must list the decoder. The adapter is
#                      tied to its decoder's embedding space, so a different decoder needs its own warm start.
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

ARM="${ARM:-azeros}"
HOURS="${HOURS:-2700}"
SITE="${SITE:-mn5}"
SEED="${SEED:-1337}"
BATCH_DURATION="${BATCH_DURATION:-64}"
MAX_ROWS="${MAX_ROWS:-16}"
FILL="${FILL:-0.8}"
MAX_DURATION="${MAX_DURATION:-30}"
if [[ "$ARM" == sift ]]; then DEFAULT_DROP_TRUNCATED=0 DEFAULT_BUDGET_FACTOR=0; else DEFAULT_DROP_TRUNCATED=1 DEFAULT_BUDGET_FACTOR=3.5; fi
DROP_TRUNCATED="${DROP_TRUNCATED:-$DEFAULT_DROP_TRUNCATED}"
REPEAT_FROM_GOLD="${REPEAT_FROM_GOLD:-0}"
ROLLOUT_LOOKAHEAD="${ROLLOUT_LOOKAHEAD:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REPLY_BUDGET_FACTOR="${REPLY_BUDGET_FACTOR:-$DEFAULT_BUDGET_FACTOR}"
ADAPTER_LR="${ADAPTER_LR:-2e-4}"
TRANSLATE_FRAC="${TRANSLATE_FRAC:-0.25}"
TARGETS_NO_GA_MT="bg,cs,da,de,el,en,es,et,fi,fr,hr,hu,it,lt,lv,nl,pl,pt,ro,sk,sl,sv"
TRANSLATE_TARGETS="${TRANSLATE_TARGETS:-$TARGETS_NO_GA_MT}"
MASTER_DTYPE="${MASTER_DTYPE:-float32}"
MASTER_DTYPE_SCOPE="${MASTER_DTYPE_SCOPE:-trainable}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20}"
export MELT_NODES="${MELT_NODES:-2}"
export MELT_GPUS_PER_NODE="${MELT_GPUS_PER_NODE:-4}"
WORLD_SIZE=$(( MELT_NODES * MELT_GPUS_PER_NODE ))

case "$ARM" in
    azeros|azeros_anchor) ;;
    ce) TRANSLATE_FRAC=0 ;;
    sift)
        TRANSLATE_FRAC=0
        [[ "$REPEAT_FROM_GOLD" != 1 ]] || die "REPEAT_FROM_GOLD=1 with ARM=sift: there are no repeat rows, the teacher replies to the bare transcript" ;;
    *) die "unknown ARM=$ARM (azeros|azeros_anchor|ce|sift)" ;;
esac
if (( ROLLOUT_LOOKAHEAD > 1 )); then
    [[ "$ARM" != ce ]] || die "ROLLOUT_LOOKAHEAD=$ROLLOUT_LOOKAHEAD with ARM=ce: the control has no teacher"
    [[ "$REPEAT_FROM_GOLD" != 1 ]] || die "ROLLOUT_LOOKAHEAD=$ROLLOUT_LOOKAHEAD with REPEAT_FROM_GOLD=1: not combinable"
    awk "BEGIN { exit !(${REPLY_BUDGET_FACTOR} > 0) }" && die "ROLLOUT_LOOKAHEAD=$ROLLOUT_LOOKAHEAD needs the fixed budget: REPLY_BUDGET_FACTOR=0, not ${REPLY_BUDGET_FACTOR}"
fi
if [[ -z "${MELT_TIME:-}" ]]; then
    [[ "${DRY_RUN:-0}" == "1" ]] || die "set MELT_TIME (HH:MM:SS); the site default is one hour"
    echo "[DRY_RUN] WARNING: MELT_TIME is unset: a real submit refuses to run without it" >&2
fi

CONFIG="projects/self-distill/sft/SD-EU24-${HOURS}h.yaml"
[[ -f "$CONFIG" ]] || die "$CONFIG not found: render it with build_sft_mix.py --sd24-hours ${HOURS}"

DEFAULT_GOLDW_RUN=GOLDW-700asr-whisperlargeF-qwen35_2bInsF-mlpT-sk5-bd150-ga1-wsd-wus0p03-lr1e3-1000h-s42-2g
WARM_START="${WARM_START:-/workspace/outputs/${DEFAULT_GOLDW_RUN}}"
INIT_TAG="${INIT_TAG:-GOLDW-final}"

# The derived schedule, in exact rational arithmetic (total_hours is a rounded decimal).
PLAN="$(python3 - "$CONFIG" "$BATCH_DURATION" "$WORLD_SIZE" "$FILL" <<'PY'
import math
import sys
from fractions import Fraction

import yaml

config, batch_duration, world_size, fill = sys.argv[1], *(Fraction(a) for a in sys.argv[2:])
with open(config) as f:
    hours = Fraction(str(yaml.safe_load(f)["data"]["train_ds"]["total_hours"]))
if hours <= 0:
    sys.exit(f"{config}: data.train_ds.total_hours is {hours}; render it with build_sft_mix.py")
print(float(hours), math.ceil(hours * 3600 / (batch_duration * world_size * fill)))
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
RUN_PREFIX="SD24-${ARM}"
[[ "$ARM" != sift ]] || RUN_PREFIX="SD-SIFT24"
export EXP_NAME="${EXP_NAME:-${RUN_PREFIX}-from-${INIT_TAG}${MIX_TAG}-${HOURS}h-bd${BATCH_DURATION}-${WORLD_SIZE}g-s${SEED}${STEPS_TAG}}"

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

EXTRAS=("${RUN_DIR_ARGS[@]}" --data.train_ds.max_duration "$MAX_DURATION" --data.train_ds.batch_size "$MAX_ROWS")
[[ "$MASTER_DTYPE" == none ]] || EXTRAS+=(--model.master_dtype "$MASTER_DTYPE" --model.master_dtype_scope "$MASTER_DTYPE_SCOPE")
[[ "$ARM" == ce ]] || EXTRAS+=(--distill.reply_budget_factor "$REPLY_BUDGET_FACTOR")
[[ "$ARM" == ce || "$DROP_TRUNCATED" != 1 ]] || EXTRAS+=(--distill.drop_truncated true)
(( ROLLOUT_LOOKAHEAD == 1 )) || EXTRAS+=(--distill.rollout_lookahead "$ROLLOUT_LOOKAHEAD")
if [[ "$ARM" != ce && "$REPEAT_FROM_GOLD" == 1 ]]; then
    EXTRAS+=(--distill.repeat_from_gold true)
    export ALIGNMENT_GAP=false  # the gap scores the teacher's own reply to the repeat instruction
fi
# The value is quoted for YAML, as launch_step2.sh quotes the templates: a bare list of ISO codes is one
# string to OmegaConf, but a single code such as "no" would become a boolean.
awk "BEGIN { exit !(${TRANSLATE_FRAC} > 0) }" && EXTRAS+=(--distill.translate_targets "\"${TRANSLATE_TARGETS}\"")

echo "[launch_sd24] ${EXP_NAME}"
echo "[launch_sd24] arm:   ${ARM}, translate_frac ${TRANSLATE_FRAC}, targets ${TRANSLATE_TARGETS}"
echo "[launch_sd24] data:  cuts up to ${MAX_DURATION} s, at most ${MAX_ROWS} per micro-batch"
BUDGET_TEXT="budget ${REPLY_BUDGET_FACTOR} x content + 32"
awk "BEGIN { exit !(${REPLY_BUDGET_FACTOR} > 0) }" || BUDGET_TEXT="fixed budget"
[[ "$ARM" == ce ]] || echo "[launch_sd24] teacher: replies up to ${MAX_NEW_TOKENS} tokens (${BUDGET_TEXT}), cut ones $( [[ "$DROP_TRUNCATED" == 1 ]] && echo dropped || echo kept )"
(( ROLLOUT_LOOKAHEAD == 1 )) || echo "[launch_sd24] lookahead: the teacher decodes ${ROLLOUT_LOOKAHEAD} micro-batches per call"
[[ "$ARM" != sift ]] || echo "[launch_sd24] sift: no instruction on either path; the teacher replies to the bare transcript, the student hears the audio alone"
[[ "$ARM" == ce || "$REPEAT_FROM_GOLD" != 1 ]] || echo "[launch_sd24] repeat rows: the gold transcript is the reply; the teacher answers the translate rows only"
echo "[launch_sd24] init:  ${WARM_START}"
echo "[launch_sd24] mix:   ${CONFIG} (${TOTAL_HOURS} h)"
echo "[launch_sd24] world: ${MELT_NODES} nodes x ${MELT_GPUS_PER_NODE} GPUs = ${WORLD_SIZE}; ${BATCH_DURATION} s per rank = $(( BATCH_DURATION * WORLD_SIZE )) s of audio per step"
echo "[launch_sd24] steps: max ${MAX_STEPS} (decay $(( MAX_STEPS / 5 ))), eval every ${EVAL_STEPS}, save every ${SAVE_STEPS}; adapter LR ${ADAPTER_LR}; seed ${SEED}; master dtype ${MASTER_DTYPE} (${MASTER_DTYPE_SCOPE})"

export ARM TRANSLATE_FRAC SITE SEED BATCH_DURATION ADAPTER_LR EVAL_SAMPLES WARM_START
export CONFIG
exec bash projects/self-distill/launch_step2.sh "${EXTRAS[@]}" "$@"
