#!/usr/bin/env python
"""Verify curope: import torch first (loads libc10/libtorch), then load curope."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--curope-pkg", required=True)
    args = parser.parse_args()

    print(f"=== verify {args.label} ===")
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.dirname(args.curope_pkg))
    sys.path.insert(0, args.curope_pkg)

    import torch  # noqa: F401
    print(f"  torch {torch.__version__} cuda avail {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")

    cls = None
    for module_name in ("croco.models.curope", "src.croco.models.curope", "curope"):
        try:
            mod = __import__(module_name, fromlist=["cuRoPE2D", "cuRoPE"])
            cls = getattr(mod, "cuRoPE2D", None) or getattr(mod, "cuRoPE", None)
            mod_file = getattr(mod, "__file__", "<no file>")
            print(f"  imported {module_name} from {mod_file}; class={cls}")
            if cls is not None:
                break
        except Exception as exc:
            print(f"  import {module_name} FAILED: {type(exc).__name__}: {str(exc)[:200]}")
    if cls is None:
        return 1

    if not torch.cuda.is_available():
        print("  no cuda, skip forward")
        return 0

    try:
        rope = cls(64).cuda()
        x = torch.randn(1, 4, 16, 64, device="cuda")
        pos = torch.zeros(1, 16, 2, dtype=torch.long, device="cuda")
        y = rope(x, pos)
        torch.cuda.synchronize()
        print(f"  forward OK shape={tuple(y.shape)} dtype={y.dtype}")
        return 0
    except Exception as exc:
        print(f"  forward FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
