#!/bin/bash
set -ex

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"

unset RAY_ADDRESS || true
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=${NUM_GPUS:-2}
REAL_CPU=${REAL_CPU:-$(nproc)}
SCRIPT_DIR="$SLIME/drug_agent"
OUTPUTS_ROOT=${OUTPUTS_ROOT:-${WD:-$GROUP_SPACE/slime_wd}/outputs}
DRUG_AGENT_DATA_ROOT=${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}
DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}
mkdir -p "$DRUG_AGENT_DATA_ROOT" "$DRUG_AGENT_RUNS_ROOT"

PROMPT_DATA=${PROMPT_DATA:-$DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl}
if [ ! -f "$PROMPT_DATA" ]; then
  echo "PROMPT_DATA not found: $PROMPT_DATA"
  exit 2
fi

NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
NUM_EPOCH=${NUM_EPOCH:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LR=${LR:-1e-5}

SAVE_DIR=${SAVE_DIR:-$DRUG_AGENT_RUNS_ROOT/Qwen3.5-0.8B_drug_sft_smoke}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
HF_CHECKPOINT=${HF_CHECKPOINT:-$VERL_DATA/Qwen3.5-0.8B}
REF_LOAD=${REF_LOAD:-$VERL_DATA/Qwen3.5-0.8B_torch_dist}
LOAD=${LOAD:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

source scripts/models/qwen3.5-0.8B.sh

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
)
if [ -n "$LOAD" ]; then
  CKPT_ARGS+=(--load "$LOAD")
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "$PROMPT_DATA"
  --input-key messages
  --metadata-key metadata
  --rollout-shuffle

  --num-epoch "$NUM_EPOCH"
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --num-cpus "$REAL_CPU" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHON_CPU_FIX_DIR}:/root/Megatron-LM/:${SLIME}:${SCRIPT_DIR:-}:${PYTHONPATH:-}\",
    \"PYTHON_CPU_COUNT\": \"${REAL_CPU}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVIDIA_VISIBLE_DEVICES\": \"${NVIDIA_VISIBLE_DEVICES:-all}\",
    \"NVIDIA_DRIVER_CAPABILITIES\": \"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\",
    \"OUTPUTS_ROOT\": \"${OUTPUTS_ROOT}\",
    \"DRUG_AGENT_DATA_ROOT\": \"${DRUG_AGENT_DATA_ROOT}\",
    \"DRUG_AGENT_RUNS_ROOT\": \"${DRUG_AGENT_RUNS_ROOT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
