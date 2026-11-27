#!/usr/bin/env bash
# fasterVLA on h6 — source from repo root.
# Deploy tree: /data/project_r/fastervla-h6 （全新目录，不碰 agibotg2_pi0.5_deploy）。
#
# Usage:
#   cd /data/project_r/fastervla-h6
#   source scripts/h6/env.sh
#
# IMPORTANT: do NOT `set -e` here (sourced into interactive shell).

FASTERVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FASTERVLA_ROOT

if [[ -x "$FASTERVLA_ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$FASTERVLA_ROOT/.venv/bin/activate"
else
  echo "WARN: $FASTERVLA_ROOT/.venv missing — run scripts/h6/bootstrap_venv.sh" >&2
fi

_OPENPI_CLIENT_SRC="$FASTERVLA_ROOT/vendor/openpi-client/src"
# 只读引用老栈 vendor；bootstrap 会拷贝到本树，不写回老栈。
_OLD_OPENPI_CLIENT_SRC="/data/project_r/agibotg2_pi0.5_deploy/vendor/openpi/packages/openpi-client/src"
if [[ -d "$_OPENPI_CLIENT_SRC/openpi_client" ]]; then
  export PYTHONPATH="${_OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d "$_OLD_OPENPI_CLIENT_SRC/openpi_client" ]]; then
  export PYTHONPATH="${_OLD_OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

export PYTHONPATH="$FASTERVLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f /home/agi/deps/acados_env.sh ]]; then
  # shellcheck disable=SC1091
  source /home/agi/deps/acados_env.sh
fi

# h6 machine defaults（来自 agibotg2_pi0.5_deploy/configs/badminton/h6_*.json）
export FASTERVLA_HOST="${FASTERVLA_HOST:-127.0.0.1}"
export FASTERVLA_PORT="${FASTERVLA_PORT:-8013}"
export FASTERVLA_ROBOT_AID="${FASTERVLA_ROBOT_AID:-G2A0205C900081}"
export FASTERVLA_SDK_SOURCE_IP="${FASTERVLA_SDK_SOURCE_IP:-10.42.1.101}"
export FASTERVLA_DISCOVERY_URI="${FASTERVLA_DISCOVERY_URI:-http://10.42.1.101:2379}"
export FASTERVLA_SDK_ROOT="${FASTERVLA_SDK_ROOT:-/home/agi/app/gdk}"
export FASTERVLA_RESULTS_DIR="${FASTERVLA_RESULTS_DIR:-$FASTERVLA_ROOT/results}"
export FASTERVLA_G2_SCRIPTS="${FASTERVLA_G2_SCRIPTS:-/data/project_r/g2_tools/scripts}"

mkdir -p "$FASTERVLA_RESULTS_DIR"

echo "[fastervla/h6] root=$FASTERVLA_ROOT host=$FASTERVLA_HOST:$FASTERVLA_PORT aid=$FASTERVLA_ROBOT_AID"
