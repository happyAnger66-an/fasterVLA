#!/usr/bin/env bash
# Execute on h7 (TTY required). Default recipe: soft 1.8× (override with CONFIG=...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/h7/env.sh"

CONFIG="${CONFIG:-$ROOT/configs/deploy/h7_soft_1p8.yaml}"
# No `exec`: keep the parent shell after Ctrl+C / exit.
fastervla-deploy run \
  --config "${CONFIG}" \
  --mode execute \
  "$@"
