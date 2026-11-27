#!/usr/bin/env bash
# 全身复位到 configs/h6_reset_pose.json（非羽毛球捡球姿态）
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/h6/run_reset_pose.sh \
  --config configs/h6_execution.json \
  --pose configs/h6_reset_pose.json \
  "$@"
