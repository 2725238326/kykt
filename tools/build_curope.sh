#!/usr/bin/env bash
# Compile croco curope CUDA extension inside a target conda env, in-place.
# Usage: build_curope.sh <env> <curope-dir>
set -euo pipefail

ENV_NAME="${1:?env}"
CUROPE_DIR="${2:?curope dir}"

echo "== build curope =="
echo "env: $ENV_NAME"
echo "dir: $CUROPE_DIR"

# Use repo-detected nvcc from /usr/local/cuda-12.6 if env doesn't ship one.
SYSTEM_CUDA="/usr/local/cuda-12.6"
if [ -d "$SYSTEM_CUDA" ]; then
    export PATH="$SYSTEM_CUDA/bin:$PATH"
    export CUDA_HOME="$SYSTEM_CUDA"
    export LD_LIBRARY_PATH="$SYSTEM_CUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

# Clean prior build outputs.
rm -rf "$CUROPE_DIR/build" "$CUROPE_DIR"/*.so "$CUROPE_DIR/__pycache__" || true

# TITAN RTX is sm_75. Pin TORCH_CUDA_ARCH_LIST so nvcc emits compatible code.
export TORCH_CUDA_ARCH_LIST="7.5"

cd "$CUROPE_DIR"
echo "running setup.py build_ext --inplace inside $ENV_NAME ..."
conda run -n "$ENV_NAME" --no-capture-output \
    python setup.py build_ext --inplace 2>&1 | tail -50

echo
echo "-- post-build artifacts --"
ls -la "$CUROPE_DIR" | head -20

echo
echo "-- verify import --"
CUROPE_DIR_ENV="$CUROPE_DIR" conda run -n "$ENV_NAME" --no-capture-output python - <<'PY'
import os, sys
curope_dir = os.environ["CUROPE_DIR_ENV"]
sys.path.insert(0, os.path.dirname(curope_dir))
sys.path.insert(0, curope_dir)
try:
    import curope
    cls = getattr(curope, "cuRoPE2D", None) or getattr(curope, "cuRoPE", None)
    print("curope import OK; class:", cls)
except Exception as exc:
    print("curope import FAILED:", type(exc).__name__, str(exc)[:240])
PY
