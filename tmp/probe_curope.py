#!/usr/bin/env python
"""Probe Align3R / CUT3R curope import + forward."""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback


def probe(env_label: str, repo_root: str, curope_pkg_path: str) -> None:
    print(f"=== {env_label} ===")
    sys.path.insert(0, repo_root)
    sys.path.insert(0, curope_pkg_path)

    try:
        import torch
        print(f"  torch {torch.__version__} cuda {torch.version.cuda} avail {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")
    except Exception as exc:
        print(f"  torch import FAIL: {exc!r}")
        return

    mod = None
    for module_name in ("croco.models.curope", "curope"):
        try:
            mod = importlib.import_module(module_name)
            print(f"  curope import OK via {module_name}")
            break
        except Exception as exc:
            print(f"  curope import via {module_name} FAIL: {type(exc).__name__}: {str(exc)[:160]}")
    if mod is None:
        return

    cuRoPE2D = getattr(mod, "cuRoPE2D", None) or getattr(mod, "cuRoPE", None)
    if cuRoPE2D is None:
        print(f"  cuRoPE2D class not found, dir={[name for name in dir(mod) if not name.startswith('_')]}")
        return
    print(f"  cuRoPE2D class: {cuRoPE2D}")

    if not torch.cuda.is_available():
        print("  skip forward (no cuda)")
        return

    try:
        rope = cuRoPE2D(64).cuda()
        x = torch.randn(1, 4, 16, 64, device="cuda")
        pos = torch.zeros(1, 16, 2, dtype=torch.long, device="cuda")
        y = rope(x, pos)
        print(f"  forward OK shape={tuple(y.shape)}")
    except Exception as exc:
        print(f"  forward FAIL: {type(exc).__name__}: {str(exc)[:240]}")
        traceback.print_exc(limit=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--curope-pkg", required=True)
    args = parser.parse_args()
    probe(args.label, args.repo, args.curope_pkg)


if __name__ == "__main__":
    main()
