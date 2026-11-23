#!/usr/bin/env bash
# Install casadi + acados (C libs + acados_template) on g2s aarch64.
# Usage (on g2s): bash scripts/install_acados_g2s.sh
set -euo pipefail

ACADOS_ROOT="${ACADOS_ROOT:-/home/agi/deps/acados}"
TERA_URL="https://github.com/acados/tera_renderer/releases/download/v0.2.1/t_renderer-v0.2.1-linux-arm64"

mkdir -p "$(dirname "$ACADOS_ROOT")"
pip3 install --user "casadi>=3.6"

if [[ ! -d "$ACADOS_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/acados/acados.git "$ACADOS_ROOT"
  git -C "$ACADOS_ROOT" submodule update --recursive --init
fi

mkdir -p "$ACADOS_ROOT/bin"
if [[ ! -x "$ACADOS_ROOT/bin/t_renderer" ]]; then
  curl -fsSL -o /tmp/t_renderer-arm64 "$TERA_URL"
  install -m 755 /tmp/t_renderer-arm64 "$ACADOS_ROOT/bin/t_renderer"
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

pip3 install --user -e "$ACADOS_ROOT/interfaces/acados_template"
# acados_template imports matplotlib; system matplotlib breaks on numpy>=2.
pip3 install --user "matplotlib>=3.9"

cat > /home/agi/deps/acados_env.sh <<EOF
export ACADOS_SOURCE_DIR=$ACADOS_ROOT
export LD_LIBRARY_PATH=$ACADOS_ROOT/lib:\${LD_LIBRARY_PATH:-}
EOF

# shellcheck disable=SC1091
source /home/agi/deps/acados_env.sh
python3 - <<'PY'
import casadi
import acados_template
print("casadi", casadi.__version__, "acados_template ok")
PY
echo "ACADOS_INSTALL_OK root=$ACADOS_ROOT"
