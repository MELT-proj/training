#!/bin/bash
#
# Ablation campaign, stage 2 (IFT / instruction fine-tuning on ASR+ST). Thin
# wrapper around launch_campaign.sh -- see that file's header and README.md
# for the axes, defaults and the two correctness rules.
#
#   CONFIG=projects/ablation-campaign/ABL-IFT-125.yaml bash projects/ablation-campaign/launch_IFT.sh
#
# NOTE on max_steps: ABL-IFT-125.yaml currently pins trainer.max_steps: 6250
# in the base config, computed (per its own comment) at world_size 2. This
# launcher does NOT override max_steps -- the base config is left as the
# authority, per the "anything invariant across arms belongs in the base
# config" rule in README.md. If an arm actually runs at world_size 8 (this
# launcher's default, matching stage 1's topology), that pinned 6250 no longer
# matches "one epoch of the 1250 h mixture" at that world_size -- it was sized
# for world_size 2. This mismatch predates this PR (issue #46: the epoch
# estimate from total_cuts was unreliable, which is why max_steps was pinned
# by hand instead of derived, before the quadratic_duration correction in
# plan_arm.py existed). It is flagged here rather than silently fixed because
# fixing it changes what stage-2 arms actually train, and that cannot be
# verified without a real GPU run, which is out of scope for this PR. This
# launcher DOES derive eval_steps/save_steps from the real world_size, same as
# MA, since periodicity is safe to correct without changing training length.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE=IFT
export CONFIG="${CONFIG:-${SCRIPT_DIR}/ABL-IFT-125.yaml}"
exec "${SCRIPT_DIR}/launch_campaign.sh" "$@"
