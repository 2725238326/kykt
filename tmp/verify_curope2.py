#!/usr/bin/env python
"""Final-mile curope verification: instantiate cuRoPE2D + forward on cuda.
Imports curope2d.py directly via importlib.util so we sidestep package layout issues."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--curope-pkg", required=True)
    args = parser.parse_args()

    print(f"=== verify {args.label} ===")
    pkg_dir = Path(args.curope_pkg)
    sys.path.insert(0, str(pkg_dir))  # allow relative `import curope`

    import torch
    print(f"  torch {torch.__version__} cuda avail {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    spec = importlib.util.spec_from_file_location("curope2d_local", pkg_dir / "curope2d.py")
    if spec is None or spec.loader is None:
        print("  cannot create spec for curope2d.py")
        return 1
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"  load curope2d.py FAILED: {type(exc).__name__}: {str(exc)[:240]}")
        return 1
    cls = getattr(module, "cuRoPE2D", None)
    print(f"  cuRoPE2D class: {cls}")
    if cls is None or not torch.cuda.is_available():
        return 0 if cls else 1

    try:
        rope = cls(64).cuda()
        x = torch.randn(1, 16, 4, 64, device="cuda").contiguous()
        pos = torch.zeros(1, 16, 2, dtype=torch.long, device="cuda").contiguous()
        y = rope(x, pos)
        torch.cuda.synchronize()
        print(f"  forward OK shape={tuple(y.shape)} dtype={y.dtype}")
        return 0
    except Exception as exc:
        print(f"  forward FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
