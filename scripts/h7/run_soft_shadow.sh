#!/usr/bin/env bash
# Soft RTC shadow on h7 (zero servo). Default: 1× + 4+4 + ZP + pchip.
#   bash scripts/h7/run_soft_shadow.sh
#   CONFIG=configs/deploy/h7_soft_1p8.yaml bash scripts/h7/run_soft_shadow.sh
#   DURATION_S=30 bash scripts/h7/run_soft_shadow.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/h7/env.sh"

CONFIG="${CONFIG:-$ROOT/configs/deploy/h7_soft_1x.yaml}"
extra=()
if [[ -n "${DURATION_S:-}" ]]; then
  extra+=(--duration-s "${DURATION_S}")
fi
# No `exec`: keep the parent shell after Ctrl+C / exit.
fastervla-deploy run \
  --config "${CONFIG}" \
  --mode shadow \
  "${extra[@]}" \
  "$@"
