#!/bin/bash
set -e

if [ -f /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh ]; then
  source /root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
else
  source /home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh
fi

cd "$SLIME"

echo "SLIME=$SLIME"
echo "GROUP_SPACE=${GROUP_SPACE:-}"
echo "VERL_DATA=${VERL_DATA:-}"

echo "---- Python/CUDA check ----"
python - <<'PY'
import os
import multiprocessing
print("os.cpu_count():", os.cpu_count())
print("multiprocessing.cpu_count():", multiprocessing.cpu_count())
try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
except Exception as exc:
    print("torch import/check failed:", repr(exc))
print("GROUP_SPACE:", os.environ.get("GROUP_SPACE"))
print("SLIME:", os.environ.get("SLIME"))
print("VERL_DATA:", os.environ.get("VERL_DATA"))
PY
