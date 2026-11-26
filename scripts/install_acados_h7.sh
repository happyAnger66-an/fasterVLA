#!/usr/bin/env bash
# Install casadi + acados (C libs + acados_template) on h7 aarch64.
# Uses fasterVLA .venv (not --user). Same BLASFEO target as g2s (A76).
#
# Usage (on h7):
#   cd /data/project_r/fastervla
#   bash scripts/install_acados_h7.sh
set -euo pipefail

FASTERVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACADOS_ROOT="${ACADOS_ROOT:-/home/agi/deps/acados}"
TERA_URL="https://github.com/acados/tera_renderer/releases/download/v0.2.1/t_renderer-v0.2.1-linux-arm64"
VENV_PY="${FASTERVLA_ROOT}/.venv/bin/python"
VENV_PIP="${FASTERVLA_ROOT}/.venv/bin/pip"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: missing $VENV_PY — run scripts/h7/bootstrap_venv.sh first" >&2
  exit 1
fi

mkdir -p "$(dirname "$ACADOS_ROOT")"
"$VENV_PIP" install -q "casadi>=3.6"

if [[ ! -d "$ACADOS_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/acados/acados.git "$ACADOS_ROOT"
  git -C "$ACADOS_ROOT" submodule update --recursive --init
fi

mkdir -p "$ACADOS_ROOT/bin"
if [[ ! -x "$ACADOS_ROOT/bin/t_renderer" ]] || ! file "$ACADOS_ROOT/bin/t_renderer" | grep -q 'ELF'; then
  # GitHub release binary often times out from CN; fall back to cargo build via rsproxy.
  if curl -fL --retry 5 --connect-timeout 20 --max-time 300 -o /tmp/t_renderer-arm64 "$TERA_URL" \
    && [[ "$(stat -c%s /tmp/t_renderer-arm64)" -ge 8000000 ]]; then
    install -m 755 /tmp/t_renderer-arm64 "$ACADOS_ROOT/bin/t_renderer"
  else
    echo "WARN: t_renderer download incomplete — building from source" >&2
    export CARGO_HOME="${CARGO_HOME:-/home/agi/deps/cargo}"
    export RUSTUP_HOME="${RUSTUP_HOME:-/home/agi/deps/rustup}"
    export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://rsproxy.cn}"
    export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://rsproxy.cn/rustup}"
    export PATH="$CARGO_HOME/bin:$PATH"
    if ! command -v cargo >/dev/null; then
      curl -fsSL https://rsproxy.cn/rustup/dist/aarch64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init-bin
      chmod +x /tmp/rustup-init-bin
      /tmp/rustup-init-bin -y --default-toolchain stable --profile minimal
      # shellcheck disable=SC1091
      source "$CARGO_HOME/env"
    fi
    mkdir -p "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<'CFG'
[source.crates-io]
replace-with = 'rsproxy-sparse'
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
CFG
    TERA_SRC=/home/agi/deps/tera_renderer
    if [[ ! -f "$TERA_SRC/Cargo.toml" ]]; then
      rm -rf "$TERA_SRC"
      git clone --depth 1 --branch v0.2.1 \
        https://ghproxy.net/https://github.com/acados/tera_renderer.git "$TERA_SRC"
    fi
    (cd "$TERA_SRC" && cargo build --release)
    install -m 755 "$TERA_SRC/target/release/t_renderer" "$ACADOS_ROOT/bin/t_renderer"
  fi
fi

if [[ ! -e "$ACADOS_ROOT/lib/libacados.so" ]]; then
  rm -rf "$ACADOS_ROOT/build"
  mkdir -p "$ACADOS_ROOT/build"
  cmake -S "$ACADOS_ROOT" -B "$ACADOS_ROOT/build" \
    -DACADOS_WITH_QPOASES=ON \
    -DBLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76 \
    -DHPIPM_TARGET=GENERIC \
    -DCMAKE_BUILD_TYPE=Release
  # armv8 kernels #include from source tree; cmake writes headers into build/.
  cp -f "$ACADOS_ROOT/build/external/blasfeo/include/blasfeo_target.h" \
    "$ACADOS_ROOT/external/blasfeo/include/blasfeo_target.h"
  cp -f "$ACADOS_ROOT/build/external/blasfeo/include/blasfeo_block_size.h" \
    "$ACADOS_ROOT/external/blasfeo/include/blasfeo_block_size.h" 2>/dev/null || true
  # Headers must exist before parallel kernel compile.
  cmake --build "$ACADOS_ROOT/build" --target blasfeo -j1
  cmake --build "$ACADOS_ROOT/build" --target install -j"$(nproc)"
fi

"$VENV_PIP" install -q -e "$ACADOS_ROOT/interfaces/acados_template"
# acados_template imports matplotlib; system matplotlib breaks on numpy>=2.
"$VENV_PIP" install -q "matplotlib>=3.9"

cat > /home/agi/deps/acados_env.sh <<EOF
export ACADOS_SOURCE_DIR=$ACADOS_ROOT
export LD_LIBRARY_PATH=$ACADOS_ROOT/lib:\${LD_LIBRARY_PATH:-}
export PATH=$ACADOS_ROOT/bin:\${PATH:-}
EOF

# shellcheck disable=SC1091
source /home/agi/deps/acados_env.sh
"$VENV_PY" - <<'PY'
import casadi
import acados_template
print("casadi", casadi.__version__, "acados_template ok")
PY
echo "ACADOS_INSTALL_OK root=$ACADOS_ROOT venv=$FASTERVLA_ROOT/.venv"
