# Drug Agent OPD

This path uses slime-native Megatron on-policy distillation on fixed offline ReAct decision states.
The current student samples the next action through SGLang; the action is never executed.

On a four-H200 worker, both the default student initialization and frozen teacher are the base
Qwen3.5-4B `torch_dist` checkpoint. This is suitable for validating the OPD pipeline, but identical
initial models provide little initial distillation signal. Set `STUDENT_LOAD` to a prior student
checkpoint and/or `OPD_TEACHER_LOAD` to a stronger compatible Qwen3.5-4B slime/Megatron checkpoint
for a meaningful learning experiment.

The default full run covers the existing 2,469 ToolRL fixed decision states once. See
[`../OFFLINE_TRAINING_POLICY.md`](../OFFLINE_TRAINING_POLICY.md).
