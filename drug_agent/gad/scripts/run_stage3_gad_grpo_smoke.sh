#!/bin/bash
set -euo pipefail
export NUM_ROLLOUT=${NUM_ROLLOUT:-2}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1}
exec bash drug_agent/gad/scripts/run_stage3_gad_grpo.sh

