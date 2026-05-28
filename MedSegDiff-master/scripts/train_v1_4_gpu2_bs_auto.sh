#!/bin/bash
# 先尝试 bs=2；若启动后短时间内 OOM/崩溃，则自动退回 bs=1 重新训练。
# 用法: bash scripts/train_v1_4_gpu2_bs_auto.sh [额外参数传给 dose_train_3d_v1_4.py]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SUFFIX="$(date +%Y%m%d_%H%M%S)"
COMMON_ARGS=(
  --epoch 600
  --val_every 50
  --save_every 50
)

try_bs() {
  local BS=$1
  shift  # 其余参数才是传给 dose_train_3d_v1_4.py 的额外 CLI
  local TAG="bs${BS}_${SUFFIX}"
  local LOG="scripts/train_v1_4_gpu2_${TAG}.log"
  echo "[auto] Trying batch_size=${BS}, log=${LOG}"

  nohup env TRAIN_BS="${BS}" bash scripts/train_v1_4_gpu2.sh \
    "${COMMON_ARGS[@]}" \
    --save_name_suffix "${TAG}" \
    "$@" \
    > "${LOG}" 2>&1 &
  local PID=$!
  echo "[auto] PID=${PID} LOG=${LOG}"

  # 等待首个 backward 或 OOM（约 3 分钟）
  local WAIT=180
  local ELAPSED=0
  while (( ELAPSED < WAIT )); do
    sleep 15
    ELAPSED=$((ELAPSED + 15))

    if ! kill -0 "${PID}" 2>/dev/null; then
      if grep -qiE 'out of memory|CUDA out of memory|OOM|CUBLAS_STATUS_ALLOC_FAILED|CUDA error' "${LOG}" 2>/dev/null; then
        echo "[auto] bs=${BS} failed (OOM/CUDA). See ${LOG}"
        return 1
      fi
      echo "[auto] bs=${BS} process exited early (non-OOM?). tail:"
      tail -20 "${LOG}" || true
      return 1
    fi

    if grep -qiE 'out of memory|CUDA out of memory|OOM|CUBLAS_STATUS_ALLOC_FAILED' "${LOG}" 2>/dev/null; then
      echo "[auto] Detected OOM in log; killing bs=${BS} ..."
      kill "${PID}" 2>/dev/null || true
      sleep 3
      pkill -f 'dose_train_3d_v1_4.py' 2>/dev/null || true
      return 1
    fi

    # 已进入正常训练（出现 Epoch 进度条）
    if grep -qE 'Epoch [0-9]+/600:.*loss=' "${LOG}" 2>/dev/null; then
      echo "[auto] bs=${BS} training looks healthy after ${ELAPSED}s."
      echo "[auto] tail -f ${LOG}"
      return 0
    fi
  done

  echo "[auto] bs=${BS} still running after ${WAIT}s — assuming OK."
  return 0
}

# 停掉已有 v1.4 训练
pkill -f 'dose_train_3d_v1_4.py' 2>/dev/null || true
sleep 3

if try_bs 2 "$@"; then
  exit 0
fi

echo "[auto] Falling back to bs=1 ..."
pkill -f 'dose_train_3d_v1_4.py' 2>/dev/null || true
sleep 3

if try_bs 1 "$@"; then
  exit 0
fi

echo "[auto] bs=1 also failed. Check logs under scripts/train_v1_4_gpu2_bs*.log"
exit 1
