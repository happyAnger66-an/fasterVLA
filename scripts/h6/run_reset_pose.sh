#!/usr/bin/env bash
# Standalone 24-DoF reset (default: read-only plan). Pass --execute to move.
set -eo pipefail
ulimit -c 0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Match h7/h6 legacy reset: app env first, then GDK libs.
if [[ -f /home/agi/app/env.sh ]]; then
  # shellcheck disable=SC1091
  source /home/agi/app/env.sh /home/agi/app >/dev/null
fi
export LD_LIBRARY_PATH="/home/agi/app/lib:/home/agi/app/lib/hal:/home/agi/app/gdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f /home/agi/app/gdk/config/app_conf.yml ]]; then
  export APP_CONF_PATH=/home/agi/app/gdk/config/app_conf.yml
fi

PY=""
for cand in \
  /data/project_r/agibotg2_pi0.5_deploy/.venv/bin/python \
  "$ROOT/.venv/bin/python" \
  /data/project_r/agibotg2_accel_deploy/.venv/bin/python
do
  if [[ -x "$cand" ]]; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "No usable python venv found for reset" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -u "$ROOT/scripts/reset_pose.py" "$@"
