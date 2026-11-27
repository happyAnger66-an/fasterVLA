#!/usr/bin/env bash
# Execute on h6 (TTY required). Default recipe: soft 1.8× (override with CONFIG=...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/h6/env.sh"

CONFIG="${CONFIG:-$ROOT/configs/deploy/h6_k1p4_qp_phase_lead2_ctrl.yaml}"
# No `exec`: keep the parent shell after Ctrl+C / exit.
fastervla-deploy run \
  --config "${CONFIG}" \
  --mode execute \
  "$@"
