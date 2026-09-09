#!/bin/bash
#
# Ablation campaign, stage 2 (IFT / instruction fine-tuning on ASR+ST). Thin
# wrapper around launch_campaign.sh -- see that file's header and README.md
# for the axes, defaults and the two correctness rules.
#
#   CONFIG=projects/ablation-campaign/ABL-IFT-700.yaml bash projects/ablation-campaign/launch_IFT.sh
#
# This launcher derives eval_steps/save_steps from the real world_size, same
# as MA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE=IFT
export CONFIG="${CONFIG:-${SCRIPT_DIR}/ABL-IFT-700.yaml}"
exec "${SCRIPT_DIR}/launch_campaign.sh" "$@"
