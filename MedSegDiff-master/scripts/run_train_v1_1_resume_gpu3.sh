#!/bin/bash
# Backward-compatible wrapper: now auto-picks any free GPU (0–3).
exec "$(dirname "$0")/run_train_v1_1_resume_auto_gpu.sh" "$@"
