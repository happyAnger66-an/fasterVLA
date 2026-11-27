#!/usr/bin/env bash
# 仅手臂 → 羽毛球捡球姿态（躯干/头/爪保持当前）
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/h6/run_reset_pose.sh \
  --config configs/badminton651/h6_89999_sync.json \
  --pose results/saved_poses/h6-20261106-150648.json \
  --arms-only \
  "$@"
