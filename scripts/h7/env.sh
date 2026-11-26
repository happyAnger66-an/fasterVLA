#!/usr/bin/env bash
# fasterVLA on h7 — source from repo root. Does NOT touch agibotg2_pi0.5_deploy.
#
# Usage:
#   cd /data/project_r/fastervla
#   source scripts/h7/env.sh
#   fastervla-deploy shadow --host 127.0.0.1 ...   # see run_shadow_example.sh
#
# IMPORTANT: do NOT `set -e` here. This file is sourced into the interactive
# shell; with `set -e`, Ctrl+C (exit 130) on fastervla-deploy would kill the
# shell and drop the SSH session.

FASTERVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FASTERVLA_ROOT

# Isolated venv for this tree only.
if [[ -x "$FASTERVLA_ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$FASTERVLA_ROOT/.venv/bin/activate"
else
  echo "WARN: $FASTERVLA_ROOT/.venv missing — run scripts/h7/bootstrap_venv.sh" >&2
fi

# openpi-client: prefer local vendor copy under this tree; else read-only path
# into old stack vendor (never write there).
_OPENPI_CLIENT_SRC="$FASTERVLA_ROOT/vendor/openpi-client/src"
_OLD_OPENPI_CLIENT_SRC="/data/project_r/agibotg2_pi0.5_deploy/vendor/openpi/packages/openpi-client/src"
if [[ -d "$_OPENPI_CLIENT_SRC/openpi_client" ]]; then
  export PYTHONPATH="${_OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d "$_OLD_OPENPI_CLIENT_SRC/openpi_client" ]]; then
  export PYTHONPATH="${_OLD_OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Editable install / src layout
export PYTHONPATH="$FASTERVLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Acados MPC (optional): install via scripts/install_acados_h7.sh
if [[ -f /home/agi/deps/acados_env.sh ]]; then
  # shellcheck disable=SC1091
  source /home/agi/deps/acados_env.sh
fi

# h7 machine defaults (override on CLI if needed)
export FASTERVLA_HOST="${FASTERVLA_HOST:-127.0.0.1}"
export FASTERVLA_PORT="${FASTERVLA_PORT:-8013}"
export FASTERVLA_ROBOT_AID="${FASTERVLA_ROBOT_AID:-G2A0205C800352}"
export FASTERVLA_SDK_SOURCE_IP="${FASTERVLA_SDK_SOURCE_IP:-192.168.112.212}"
export FASTERVLA_DISCOVERY_URI="${FASTERVLA_DISCOVERY_URI:-http://192.168.112.212:2379}"
export FASTERVLA_SDK_ROOT="${FASTERVLA_SDK_ROOT:-/home/agi/app/gdk}"
export FASTERVLA_RESULTS_DIR="${FASTERVLA_RESULTS_DIR:-$FASTERVLA_ROOT/results}"
# URDF FK helper for report --urdf (grip_close EE mm); sibling of fastervla on h7.
export FASTERVLA_G2_SCRIPTS="${FASTERVLA_G2_SCRIPTS:-/data/project_r/g2_tools/scripts}"

mkdir -p "$FASTERVLA_RESULTS_DIR"

echo "[fastervla/h7] root=$FASTERVLA_ROOT host=$FASTERVLA_HOST:$FASTERVLA_PORT aid=$FASTERVLA_ROBOT_AID"
