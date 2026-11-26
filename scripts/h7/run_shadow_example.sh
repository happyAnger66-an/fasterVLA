#!/usr/bin/env bash
# Shadow on h7 using local inference :8013. Zero servo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/h7/env.sh"

# No `exec`: keep the parent shell after Ctrl+C / exit.
fastervla-deploy shadow \
  --host "${FASTERVLA_HOST}" \
  --port "${FASTERVLA_PORT}" \
  --sdk-root "${FASTERVLA_SDK_ROOT}" \
  --robot-aid "${FASTERVLA_ROBOT_AID}" \
  --sdk-source-ip "${FASTERVLA_SDK_SOURCE_IP}" \
  --discovery-uri "${FASTERVLA_DISCOVERY_URI}" \
  --results-dir "${FASTERVLA_RESULTS_DIR}" \
  --duration-s "${DURATION_S:-30}" \
  --infer-period-s "${INFER_PERIOD_S:-0.2}" \
  "$@"
