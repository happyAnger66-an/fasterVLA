#!/usr/bin/env bash
# Create /data/project_r/fastervla/.venv and install deploy-client deps.
# Safe: never modifies /data/project_r/agibotg2_pi0.5_deploy.
set -euo pipefail

FASTERVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FASTERVLA_ROOT"

if [[ -d .venv ]]; then
  echo "[bootstrap] .venv already exists — skip create (use --recreate to wipe)"
  if [[ "${1:-}" == "--recreate" ]]; then
    rm -rf .venv
  else
    :
  fi
fi

if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python python3.12
  elif command -v virtualenv >/dev/null 2>&1; then
    virtualenv -p python3.12 .venv
  else
    python3 -m venv .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Prefer uv pip (uv venv often has no ensurepip).
if command -v uv >/dev/null 2>&1; then
  uv pip install -U pip setuptools wheel
  # typing_extensions / msgpack required by vendored openpi-client
  uv pip install -e '.[deploy-client,dev]'
else
  python -m ensurepip --upgrade || true
  python -m pip install -U pip setuptools wheel
  python -m pip install -e '.[deploy-client,dev]'
fi


# openpi-client: copy read-only vendor into this tree if missing.
VENDOR_DST="$FASTERVLA_ROOT/vendor/openpi-client"
VENDOR_SRC="/data/project_r/agibotg2_pi0.5_deploy/vendor/openpi/packages/openpi-client"
if [[ ! -d "$VENDOR_DST/src/openpi_client" ]]; then
  if [[ -d "$VENDOR_SRC/src/openpi_client" ]]; then
    mkdir -p "$FASTERVLA_ROOT/vendor"
    cp -a "$VENDOR_SRC" "$VENDOR_DST"
    echo "[bootstrap] copied openpi-client → vendor/openpi-client (local copy)"
  else
    echo "[bootstrap] WARN: openpi-client vendor not found at $VENDOR_SRC" >&2
  fi
fi

python - <<'PY'
import importlib
for m in ("numpy", "yaml", "websockets", "scipy"):
    importlib.import_module(m)
    print("ok", m)
print("fastervla import…")
import fastervla.deploy.cli  # noqa: F401
print("ok fastervla.deploy.cli")
PY

echo "[bootstrap] done → $FASTERVLA_ROOT/.venv"
