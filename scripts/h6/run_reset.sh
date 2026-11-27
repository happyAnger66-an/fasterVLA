#!/usr/bin/env bash
# 羽毛球捡球初始姿态复位（与 h7 常用命令等价）:
#   bash .../run_reset_pose.sh \
#     --config configs/badminton651/h6_89999_sync.json \
#     --pose results/saved_poses/h6-20261106-150648.json [--execute]
# 默认只读规划；加 --execute 才下发（需 TTY）。
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/h6/run_reset_pose.sh \
  --config configs/badminton651/h6_89999_sync.json \
  --pose results/saved_poses/h6-20261106-150648.json \
  "$@"
