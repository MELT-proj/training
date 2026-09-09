#!/bin/bash
#
# Ablation campaign, stage 1 (MA / modality alignment). Thin wrapper around
# launch_campaign.sh -- see that file's header and README.md for the axes,
# defaults and the two correctness rules. Replaces
# launch_MA_llama32-1b-instruct_mn5.sh (125 h arm) and
# launch_MA_700_llama32-1b_mn5.sh (700 h arm): both are now
#
#   CONFIG=projects/ablation-campaign/ABL-MA-125-asr.yaml bash projects/ablation-campaign/launch_MA.sh
#   CONFIG=projects/ablation-campaign/ABL-MA-700-asr.yaml bash projects/ablation-campaign/launch_MA.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE=MA
export CONFIG="${CONFIG:-${SCRIPT_DIR}/ABL-MA-125-asr.yaml}"
exec "${SCRIPT_DIR}/launch_campaign.sh" "$@"
