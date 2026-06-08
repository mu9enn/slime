#!/bin/bash
set -euo pipefail
if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh; else source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh; fi
cd "$SLIME"
source drug_agent/scripts/offline_training_env.sh
PROMPT_DATA=${PROMPT_DATA:-$VERL_DATA/slime_drug_agent_data/toolrl/mcp_sft_all.toolrl_steps.jsonl}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-scripts/models/qwen3.5-4B.sh}
HF_CHECKPOINT=${HF_CHECKPOINT:-$VERL_DATA/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-$VERL_DATA/Qwen3.5-4B_torch_dist}
STUDENT_LOAD=${STUDENT_LOAD:-$REF_LOAD}
OPD_TEACHER_LOAD=${OPD_TEACHER_LOAD:-$REF_LOAD}
RUN_NAME=${RUN_NAME:-Qwen3.5-4B_opd_$(date +%Y%m%d_%H%M%S)}
SAVE_DIR=${SAVE_DIR:-$VERL_DATA/slime_drug_agent_runs/$RUN_NAME}
NUM_GPUS=${NUM_GPUS:-4}; TP=${TENSOR_MODEL_PARALLEL_SIZE:-4}; ROLLOUT_TP=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}; RBS=${ROLLOUT_BATCH_SIZE:-2}; N_SAMPLES=${N_SAMPLES_PER_PROMPT:-2}; GBS=${GLOBAL_BATCH_SIZE:-4}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-6144}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-6656}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
for path in "$PROMPT_DATA" "$MODEL_ARGS_FILE" "$HF_CHECKPOINT" "$REF_LOAD" "$OPD_TEACHER_LOAD"; do
  [ -e "$path" ] || { echo "Required OPD input does not exist: $path" >&2; exit 2; }
done
if [ ! -f "$STUDENT_LOAD/latest_checkpointed_iteration.txt" ]; then
  echo "STUDENT_LOAD is not a valid completed slime checkpoint: $STUDENT_LOAD" >&2
  echo "Valid checkpoint candidates under \$VERL_DATA/slime_drug_agent_runs:" >&2
  find "$VERL_DATA/slime_drug_agent_runs" -maxdepth 2 -type f -name latest_checkpointed_iteration.txt \
    -printf '  %h\n' 2>/dev/null | sort >&2 || true
  exit 2
fi
[ $((NUM_GPUS % TP)) -eq 0 ] || { echo "NUM_GPUS must be divisible by TP" >&2; exit 2; }
TOTAL_SAMPLES=$((RBS * N_SAMPLES)); [ "$TOTAL_SAMPLES" -ge "$GBS" ] && [ $((TOTAL_SAMPLES % GBS)) -eq 0 ] || { echo "RBS*N_SAMPLES must be >= and divisible by GBS" >&2; exit 2; }
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then unset PYTORCH_CUDA_ALLOC_CONF; fi
if [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then unset PYTORCH_ALLOC_CONF; fi
source "$MODEL_ARGS_FILE"
unset RAY_ADDRESS || true; ray stop --force 2>/dev/null || true; pkill -9 sglang 2>/dev/null || true; pkill -9 ray 2>/dev/null || true
ray start --head --node-ip-address=127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0
RUNTIME_ENV="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${PYTHONPATH:-}\",\"DRUG_AGENT_TRAINING_OFFLINE\":\"1\",\"DRUG_AGENT_ALLOW_TOOL_ENV\":\"0\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}"
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV" -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --colocate "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD" --load "$STUDENT_LOAD" --finetune --no-load-optim --no-load-rng --start-rollout-id 0 --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-1}" \
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata --apply-chat-template --rollout-shuffle \
  --custom-rm-path drug_agent.gad.negative_cache.zero_reward --advantage-estimator grpo --use-opd --opd-type megatron --opd-kl-coef "${OPD_KL_COEF:-1.0}" --opd-teacher-load "$OPD_TEACHER_LOAD" \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$RBS" --n-samples-per-prompt "$N_SAMPLES" --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN" --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}" --rollout-num-gpus-per-engine "$ROLLOUT_TP" --global-batch-size "$GBS" --balance-data \
  --tensor-model-parallel-size "$TP" --sequence-parallel --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
