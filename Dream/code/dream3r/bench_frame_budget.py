"""
Frame budget profiler for Dream3R.

Measures per-module latency and reports p50/p95/p99 statistics.
Uses CUDA events for GPU timing, time.perf_counter for CPU.

Usage:
    python -m dream3r.bench_frame_budget --preset small --n-windows 50
"""

import argparse
import time
from typing import Dict, List

import torch
import numpy as np


def profile_model(model, x, regime, n_windows: int = 50,
                  device: torch.device = torch.device("cpu"),
                  warmup: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Profile Dream3R forward pass timing.

    Returns dict of module_name -> {p50, p95, p99, mean, min, max} in ms.
    """
    model.eval()
    model.profile = True

    use_cuda = device.type == "cuda"
    all_timings: Dict[str, List[float]] = {}
    total_times: List[float] = []

    with torch.no_grad():
        prev_mem = None
        prev_slots = None

        for i in range(warmup + n_windows):
            if use_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            out = model(x, regime,
                        prev_memory_state=prev_mem,
                        prev_object_slots=prev_slots,
                        timestep=i)

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                total_ms = start_event.elapsed_time(end_event)
            else:
                total_ms = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                total_times.append(total_ms)
                if "timings" in out:
                    for k, v in out["timings"].items():
                        if k not in all_timings:
                            all_timings[k] = []
                        all_timings[k].append(v)

            if "latent_state_tokens" in out:
                prev_mem = out["latent_state_tokens"].detach()
            elif "latent_state" in out:
                prev_mem = out["latent_state"].detach()
            if "object_track_set" in out:
                prev_slots = out["object_track_set"].detach()

    model.profile = False

    results = {}
    for name, times in all_timings.items():
        arr = np.array(times)
        results[name] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    total_arr = np.array(total_times)
    results["total"] = {
        "p50": float(np.percentile(total_arr, 50)),
        "p95": float(np.percentile(total_arr, 95)),
        "p99": float(np.percentile(total_arr, 99)),
        "mean": float(total_arr.mean()),
        "min": float(total_arr.min()),
        "max": float(total_arr.max()),
    }

    return results


def print_report(results: Dict[str, Dict[str, float]], target_ms: float = 50.0):
    print(f"\n{'Module':<20} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'budget':>8}")
    print("-" * 72)

    total = results.get("total", {})
    for name, stats in sorted(results.items()):
        if name == "total":
            continue
        pct = stats["mean"] / total.get("mean", 1) * 100
        print(f"  {name:<18} {stats['p50']:>7.2f} {stats['p95']:>7.2f} "
              f"{stats['p99']:>7.2f} {stats['mean']:>7.2f} {pct:>6.1f}%")

    print("-" * 72)
    if total:
        status = "OK" if total["p95"] <= target_ms else "OVER"
        print(f"  {'TOTAL':<18} {total['p50']:>7.2f} {total['p95']:>7.2f} "
              f"{total['p99']:>7.2f} {total['mean']:>7.2f}  [{status}]")
        print(f"\n  Target: {target_ms:.0f} ms/frame | p95: {total['p95']:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small")
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-ms", type=float, default=50.0)
    args = parser.parse_args()

    from dream3r.model import build_dream3r

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling Dream3R [{args.preset}] on {device}")

    model = build_dream3r(args.preset).to(device)
    model.profile = True

    B, N, P, D = args.batch_size, 4, 196, 768
    x = torch.randn(B, N, P, D, device=device)
    regime = torch.softmax(torch.randn(B, 5, device=device), dim=-1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    results = profile_model(model, x, regime, n_windows=args.n_windows,
                            device=device)
    print_report(results, target_ms=args.target_ms)


if __name__ == "__main__":
    main()
