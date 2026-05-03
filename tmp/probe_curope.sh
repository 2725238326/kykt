#!/usr/bin/env bash
# Probe Align3R and CUT3R curope/CUDA state on the remote server.
set -uo pipefail

echo "=== System CUDA ==="
which nvcc 2>/dev/null
nvcc --version 2>/dev/null | sed -n '4p'
echo

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1
echo

probe_env () {
    local env_name="$1"
    local repo_root="$2"
    local curope_pkg_path="$3"
    echo "=== ${env_name} env ==="
    conda run -n "$env_name" --no-capture-output python <<PY 2>&1 | sed 's/^/  /'
import sys
try:
    import torch
    print("python", sys.version.split()[0])
    print("torch", torch.__version__)
    print("torch.cuda", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print("device capability", cap)
        print("device name", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import FAILED:", repr(exc))
PY

    echo "  --- ${env_name} curope import + forward ---"
    conda run -n "$env_name" --no-capture-output python <<PY 2>&1 | sed 's/^/  /'
import sys, importlib
sys.path.insert(0, ${repo_root!r})
sys.path.insert(0, ${curope_pkg_path!r})
try:
    mod = importlib.import_module("croco.models.curope")
    print("imported via croco.models.curope")
except Exception as exc:
    print("croco import failed:", repr(exc)[:200])
    try:
        import curope as mod  # type: ignore
        print("imported via local curope module")
    except Exception as exc2:
        print("local curope import failed:", repr(exc2)[:200])
        sys.exit(0)

cuRoPE2D = getattr(mod, "cuRoPE2D", None) or getattr(mod, "cuRoPE", None)
if cuRoPE2D is None:
    print("cuRoPE2D class not found in module:", dir(mod))
    sys.exit(0)
print("cuRoPE2D class:", cuRoPE2D)

import torch
if not torch.cuda.is_available():
    print("CUDA not available, skipping forward test")
    sys.exit(0)
try:
    rope = cuRoPE2D(64).cuda()
    x = torch.randn(1, 4, 16, 64, device="cuda")
    pos = torch.zeros(1, 16, 2, dtype=torch.long, device="cuda")
    y = rope(x, pos)
    print("forward OK shape", tuple(y.shape))
except Exception as exc:
    print("forward FAILED:", repr(exc)[:300])
PY
    echo
}

probe_env align3r /hdd3/kykt26/code/align3r /hdd3/kykt26/code/align3r/croco/models/curope
probe_env cut3r   /hdd3/kykt26/code/cut3r   /hdd3/kykt26/code/cut3r/src/croco/models/curope

echo "=== Build artifact survey ==="
echo "-- align3r curope dir --"
ls -la /hdd3/kykt26/code/align3r/croco/models/curope 2>&1 | sed 's/^/  /' | head -20
echo "-- cut3r curope dir --"
ls -la /hdd3/kykt26/code/cut3r/src/croco/models/curope 2>&1 | sed 's/^/  /' | head -20
echo "-- cut3r build dir (if any) --"
ls -la /hdd3/kykt26/code/cut3r/src/croco/models/curope/build 2>&1 | sed 's/^/  /' | head -20
