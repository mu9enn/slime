# Slime Drug Agent 迁移与分阶段训练手册（2026-06-01）

本手册记录了从本次对话开始到当前版本，在 **slime 原生框架** 上为 `drug_agent` 落地的做法、原理和运行方式。

## 0. 当前临时约定（2026-06-01）

- 调试模型路径策略：
  - 默认：`$DATA/Qwen3.5-122B-A10B`
  - 若默认路径不存在，自动 fallback：`$VERL_DATA/Qwen3.5-27B`
- 在 GPU worker 内，路径前缀统一使用 `/root/slime_sxy/...`，不要写 `/home/sunxiangyu/...`。
- 训练默认使用严格 rollout：
  - `DRUG_AGENT_ROLLOUT_MODE=train_strict`
  - `DRUG_AGENT_ALLOW_PARSE_RECOVERY=0`
- `debug_one_task.py` 允许宽松恢复，仅用于门禁/诊断，不用于训练 reward。
- 可先设置：

```bash
export DEBUG_MODEL_PATH=${DEBUG_MODEL_PATH:-$DATA/Qwen3.5-122B-A10B}
if [ ! -d "$DEBUG_MODEL_PATH" ]; then
  export DEBUG_MODEL_PATH=$VERL_DATA/Qwen3.5-27B
fi
```

## 1. 目标与原则

### 1.1 目标

在不改 slime core 的前提下，在 `$SLIME/drug_agent` 目录内实现：

- 数据检查与转换（`pipelined_data -> slime`）
- MCP tool 驱动的 agent rollout
- 自定义 reward
- 轨迹可审计日志
- 可运行的 SFT / PPO / GRPO 训练脚本

### 1.2 原则（已执行）

- **slime-native first**：优先 `train.py` + `custom_generate` + `custom_rm` + `custom_rollout_log`
- **不迁移 verl trainer 主干**：不搬 EnvManager / VectorEnv / trainer adapter
- **插件化边界**：新增逻辑集中在 `$SLIME/drug_agent`

## 2. 当前实现总览

## 2.1 已实现目录

```text
drug_agent/
  protocol/
  data/
  tools/
  rollout/
  scripts/
  tools_debug/
```

## 2.2 关键实现说明

- `protocol/action_parser.py`
  - 严格 JSON 协议
  - 拒绝 markdown fenced JSON / XML / 包裹文本
- `data/inspect_pipelined_data.py`
  - 扫描 `ac/pf/vs`
  - 输出 schema report + join 覆盖统计
- `data/convert_pipelined_to_slime_grpo.py`
  - Hybrid 联表（raw + sft_outputs_answer_hit + usage_summary）
  - 输出 `grpo/ac,pf,vs,mixed.jsonl` + `skipped_report.jsonl`
- `data/convert_pipelined_to_slime_sft.py`
  - answer_hit_only 数据转换
  - assistant action 归一化为裸 JSON
- `tools/*`
  - MCP client / executor / registry
  - allowlist 默认开启
- `rollout/generate_with_drug_agent.py`
  - slime custom_generate 多轮工具调用循环
- `rollout/reward_func.py`
  - async RM，返回 dict（含 `score` 与组件）
- `rollout/trajectory_logger.py`
  - 写 `trajectories.jsonl`
  - 注入 `action_valid_rate/tool_success_rate/final_success_rate`

## 3. 数据状态（你已复现成功）

你在 worker 上已跑通：

- inspect 成功：`total_files=169`
- GRPO 转换成功：`out_mixed=125`
- SFT 转换成功：`out_mixed=125`

这说明第一阶段的数据链路已经成立。

## 4. 训练阶段设计（SFT -> PPO -> GRPO）

## 4.1 为什么按这个顺序

- SFT：先让模型稳定学会 action 协议与多轮格式
- PPO：在已有行为基础上做在线策略优化（较稳）
- GRPO：再切换到 group-based 优势估计做强化

## 4.2 脚本增强（本次已更新）

以下脚本现已支持：

- `HF_CHECKPOINT`（默认 `$VERL_DATA/Qwen3.5-0.8B`）
- `REF_LOAD`（默认 `$VERL_DATA/Qwen3.5-0.8B_torch_dist`）
- `LOAD`（默认空）

可用于阶段串联：

- PPO 从 SFT 产物继续：`LOAD=$SFT_SAVE`
- GRPO 从 PPO 产物继续：`LOAD=$PPO_SAVE`

## 5. 一套可直接执行的分阶段命令

先初始化环境：

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd $SLIME
```

可选：确认数据文件存在

```bash
export OUTPUTS_ROOT=${OUTPUTS_ROOT:-$WD/outputs}
export DRUG_AGENT_DATA_ROOT=${DRUG_AGENT_DATA_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_data}
export DRUG_AGENT_RUNS_ROOT=${DRUG_AGENT_RUNS_ROOT:-$OUTPUTS_ROOT/slime_drug_agent_runs}

ls -lh $DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl
ls -lh $DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl
```

定义实验目录：

```bash
export RUN_TAG=drug_agent_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=$DRUG_AGENT_RUNS_ROOT/$RUN_TAG
mkdir -p $RUN_ROOT

export SFT_SAVE=$RUN_ROOT/ckpt_sft
export PPO_SAVE=$RUN_ROOT/ckpt_ppo
export GRPO_SAVE=$RUN_ROOT/ckpt_grpo
```

### Stage A: SFT smoke

```bash
NUM_GPUS=2 \
PROMPT_DATA=$DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl \
SAVE_DIR=$SFT_SAVE \
SAVE_INTERVAL=1 \
DRUG_AGENT_RUN_NAME=${RUN_TAG}_sft \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_sft_smoke.sh \
  2>&1 | tee $RUN_ROOT/sft.log
```

SFT 前先做数据校验（强烈建议）：

```bash
python drug_agent/data/validate_sft_messages.py \
  --input $DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl \
  --tokenizer $VERL_DATA/Qwen3.5-0.8B
```

### Stage B: PPO smoke（从 SFT 继续）

```bash
NUM_GPUS=2 \
PROMPT_DATA=$DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl \
LOAD=$SFT_SAVE \
SAVE_DIR=$PPO_SAVE \
SAVE_INTERVAL=1 \
DRUG_AGENT_RUN_NAME=${RUN_TAG}_ppo \
MOLCLAW_SCP_SERVER_URL=$MOLCLAW_SCP_SERVER_URL \
MOLCLAW_SCP_API_KEY=$MOLCLAW_SCP_API_KEY \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_ppo_smoke.sh \
  2>&1 | tee $RUN_ROOT/ppo.log
```

### Stage C: GRPO smoke（从 PPO 继续）

```bash
NUM_GPUS=2 \
PROMPT_DATA=$DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl \
LOAD=$PPO_SAVE \
SAVE_DIR=$GRPO_SAVE \
SAVE_INTERVAL=1 \
DRUG_AGENT_RUN_NAME=${RUN_TAG}_grpo_smoke \
MOLCLAW_SCP_SERVER_URL=$MOLCLAW_SCP_SERVER_URL \
MOLCLAW_SCP_API_KEY=$MOLCLAW_SCP_API_KEY \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_grpo_smoke.sh \
  2>&1 | tee $RUN_ROOT/grpo_smoke.log
```

### Stage D: GRPO learn（可选放大）

```bash
NUM_GPUS=2 \
NUM_ROLLOUT=10 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES_PER_PROMPT=4 \
GLOBAL_BATCH_SIZE=16 \
PROMPT_DATA=$DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl \
LOAD=$GRPO_SAVE \
SAVE_DIR=$RUN_ROOT/ckpt_grpo_learn \
SAVE_INTERVAL=5 \
DRUG_AGENT_RUN_NAME=${RUN_TAG}_grpo_learn \
MOLCLAW_SCP_SERVER_URL=$MOLCLAW_SCP_SERVER_URL \
MOLCLAW_SCP_API_KEY=$MOLCLAW_SCP_API_KEY \
bash drug_agent/scripts/run_qwen3_5_0_8b_drug_grpo_learn.sh \
  2>&1 | tee $RUN_ROOT/grpo_learn.log
```

## 6. 关键产物与验收点

- checkpoint：`$SFT_SAVE/$PPO_SAVE/$GRPO_SAVE`
- trajectory：`$DRUG_AGENT_RUNS_ROOT/<run_name>/trajectories.jsonl`
- 日志指标（rollout extra metrics）：
  - `action_valid_rate`
  - `strict_success_rate`
  - `recovery_success_rate`
  - `tool_success_rate`
  - `final_success_rate`

## 7. 常见问题与排障

### 7.1 `MOLCLAW_SCP_SERVER_URL is missing`

未导出 MCP 环境变量。至少需要：

```bash
export MOLCLAW_SCP_SERVER_URL=...
export MOLCLAW_SCP_API_KEY=...
```

### 7.2 `client_loop: send disconnect: Broken pipe`

通常是 SSH 会话断开，不一定代表训练失败。
建议：

- 用 `tmux`/`screen` 跑命令
- 全部命令加 `2>&1 | tee ...log`
- 断线后回连查看日志文件和 ray job 状态

### 7.3 `PROMPT_DATA not found`

先确认转换输出：

```bash
ls -lh $DRUG_AGENT_DATA_ROOT/grpo/mixed.jsonl
ls -lh $DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl
```

### 7.4 SFT 报错 `No user query found in messages`

这是 Qwen3.5 chat template 的约束触发，不是 CUDA/Ray 崩溃。修复策略已落在 `convert_pipelined_to_slime_sft.py`：  
1. 非 `system/user/assistant` 角色统一转为 `user`；  
2. `content` 统一字符串化；  
3. 保证第一条非 system 消息是 user；  
4. 如果所有 user 都是 `<tool_response>...</tool_response>`，自动插入一个种子 user query。  

修复后必须跑：

```bash
python drug_agent/data/validate_sft_messages.py \
  --input $DRUG_AGENT_DATA_ROOT/sft/mixed.jsonl \
  --tokenizer $VERL_DATA/Qwen3.5-0.8B
```

## 8. 下一步建议

- 先跑一轮最小 SFT smoke，确认 checkpoint 与 loss
- 再跑 PPO smoke（LOAD=SFT）验证在线工具链稳定性
- 再转 GRPO smoke，最后放大到 GRPO learn

## 9. 门禁调试（不启动训练）

### 9.1 MCP 工具门禁

```bash
python drug_agent/tools_debug/debug_mcp_tools.py --list-tools
python drug_agent/tools_debug/debug_mcp_tools.py \
  --tool is_valid_smiles \
  --args '{"smiles_list":["CCO"]}'
```

说明：
- 门禁阶段不做工具参数 alias 自动修补（例如不把 `smiles` 自动改写成 `smiles_list`）。
- 模型需要学习真实 MCP tool schema，schema 不匹配应显式暴露。
- 本调试脚本属于 permissive debug，允许 parse recovery；训练脚本默认 strict，不允许无惩罚恢复。

### 9.2 轨迹重放门禁

```bash
python drug_agent/tools_debug/debug_replay_trajectory.py \
  --input-jsonl $VERL_DATA/slime_drug_agent_data/sft/mixed.jsonl \
  --index 0 \
  --max-tool-calls 3 \
  --run-name gate_replay_$(date +%Y%m%d_%H%M%S)
```

### 9.3 单任务在线调试门禁

先做 SGLang launcher 探测 dry-run（122B 默认，自动 fallback 27B）：

```bash
source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
cd $SLIME

export DEBUG_MODEL_PATH=${DEBUG_MODEL_PATH:-$DATA/Qwen3.5-122B-A10B}
if [ ! -d "$DEBUG_MODEL_PATH" ]; then
  export DEBUG_MODEL_PATH=$VERL_DATA/Qwen3.5-27B
fi

python drug_agent/tools_debug/debug_sglang_launch.py \
  --model-path "$DEBUG_MODEL_PATH" \
  --tp-size 2 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.80 \
  --dry-run
```

非 dry-run 默认会后台拉起 SGLang，并返回 `pid` 与 `process_alive`：

```bash
python drug_agent/tools_debug/debug_sglang_launch.py \
  --model-path "$DEBUG_MODEL_PATH" \
  --tp-size 2 \
  --port 30000 \
  --context-length 8192 \
  --mem-fraction-static 0.80
```

如果只想验证启动链路后自动回收进程，追加 `--auto-stop`。

然后使用外部 SGLang 服务跑单任务调试：

```bash
python drug_agent/tools_debug/debug_one_task.py \
  --input-jsonl $VERL_DATA/slime_drug_agent_data/grpo/mixed.jsonl \
  --index 0 \
  --sglang-base-url http://127.0.0.1:30000 \
  --max-steps 1 \
  --temperature 0.2 \
  --max-new-tokens 1024 \
  --run-name gate_27b_step1_$(date +%Y%m%d_%H%M%S)
```

补充：
- 122B/27B 仅用于 rollout debug / teacher，不用于当前 actor 训练。
