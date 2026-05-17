#!/usr/bin/env python3
"""
Cross-model comparison smoke test.

Runs all validated models on a unified input sample via SSH dispatch,
collects timing + artifact summaries, and writes a comparison matrix.

Usage (from vision_ui/):
    python scripts/cross_model_smoke.py
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── SSH helpers ──────────────────────────────────────────────────────────────

SSH_ALIAS = "KYKT-UI"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]
REMOTE_JOBS = "/hdd3/kykt26/jobs"
REMOTE_RUNNERS = "/hdd3/kykt26/runners"
REMOTE_CODE = "/hdd3/kykt26/code"
SAMPLE_SOURCE = f"{REMOTE_CODE}/cut3r/examples/001"

WINDOWS_NO_WINDOW = 0x08000000


def ssh(cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    full = ["ssh", "-T", *SSH_OPTS, SSH_ALIAS, f"bash -lc {shlex.quote(cmd)}"]
    opts: dict = {"text": True, "encoding": "utf-8", "errors": "replace",
                  "capture_output": True, "timeout": timeout, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        opts["creationflags"] = WINDOWS_NO_WINDOW
    return subprocess.run(full, **opts)


def scp_up(local: Path, remote: str) -> None:
    subprocess.run(
        ["scp", *SSH_OPTS, str(local), f"{SSH_ALIAS}:{remote}"],
        check=True, capture_output=True, text=True,
        creationflags=WINDOWS_NO_WINDOW if os.name == "nt" else 0,
    )


# ── Model configs ───────────────────────────────────────────────────────────

@dataclass
class ModelRun:
    key: str
    label: str
    env: str
    repo: str
    runner: str
    build_cmd: str = ""           # built at runtime
    status: str = "pending"
    elapsed: float = 0.0
    artifact_count: int = 0
    error: str = ""
    scene_meta: dict = field(default_factory=dict)


MODELS: list[ModelRun] = [
    ModelRun("dust3r", "DUSt3R", "dust3r",
             f"{REMOTE_CODE}/dust3r-main",
             "dust3r_runner.py"),
    ModelRun("mast3r", "MASt3R", "mast3r",
             f"{REMOTE_CODE}/mast3r",
             "mast3r_runner.py"),
    ModelRun("monst3r", "MonST3R", "monst3r",
             f"{REMOTE_CODE}/monst3r",
             "monst3r_runner.py"),
    ModelRun("spann3r", "Spann3R", "spann3r",
             f"{REMOTE_CODE}/spann3r",
             "spann3r_runner.py"),
    ModelRun("fast3r", "Fast3R", "fast3r",
             f"{REMOTE_CODE}/fast3r",
             "fast3r_runner.py"),
    ModelRun("cut3r", "CUT3R", "cut3r",
             f"{REMOTE_CODE}/cut3r",
             "cut3r_runner.py"),
]


# ── Runner command builders ─────────────────────────────────────────────────

def _cmd_dust3r(m: ModelRun, job_dir: str) -> str:
    model_path = "/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} --model {model_path} --image-size 512 "
        f"2>&1"
    )

def _cmd_mast3r(m: ModelRun, job_dir: str) -> str:
    model_path = f"{REMOTE_CODE}/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} --model {model_path} --image-size 512 "
        f"2>&1"
    )

def _cmd_monst3r(m: ModelRun, job_dir: str) -> str:
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} "
        f"2>&1"
    )

def _cmd_spann3r(m: ModelRun, job_dir: str) -> str:
    ckpt = f"{REMOTE_CODE}/spann3r/checkpoints/spann3r.pth"
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} --checkpoint {ckpt} "
        f"2>&1"
    )

def _cmd_fast3r(m: ModelRun, job_dir: str) -> str:
    ckpt_dir = "/hdd3/kykt26/models/fast3r/Fast3R_ViT_Large_512"
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} --checkpoint-dir {ckpt_dir} "
        f"--image-size 512 --max-points 250000 --attention-backend pytorch_naive "
        f"2>&1"
    )

def _cmd_cut3r(m: ModelRun, job_dir: str) -> str:
    model_path = f"{REMOTE_CODE}/cut3r/src/cut3r_512_dpt_4_64.pth"
    return (
        f"cd {m.repo} && conda run --no-capture-output -n {m.env} "
        f"python -u {REMOTE_RUNNERS}/{m.runner} "
        f"--job-dir {job_dir} --repo {m.repo} --model-path {model_path} "
        f"--size 512 --vis-threshold 1.5 --max-frames 48 "
        f"2>&1"
    )

CMD_BUILDERS = {
    "dust3r": _cmd_dust3r,
    "mast3r": _cmd_mast3r,
    "monst3r": _cmd_monst3r,
    "spann3r": _cmd_spann3r,
    "fast3r": _cmd_fast3r,
    "cut3r": _cmd_cut3r,
}


# ── Main logic ──────────────────────────────────────────────────────────────

def upload_runners() -> None:
    """Upload all runner scripts to the server."""
    runners_dir = ROOT / "runners"
    for m in MODELS:
        local = runners_dir / m.runner
        if local.exists():
            print(f"  Uploading {m.runner}...", end=" ", flush=True)
            scp_up(local, f"{REMOTE_RUNNERS}/{m.runner}")
            print("OK")


def prepare_job(m: ModelRun) -> str:
    """Create remote job dir and copy unified sample images."""
    job_dir = f"{REMOTE_JOBS}/xmodel_{m.key}"
    ssh(f"rm -rf {job_dir} && mkdir -p {job_dir}/input {job_dir}/output {job_dir}/logs")
    ssh(f"cp {SAMPLE_SOURCE}/*.jpg {job_dir}/input/")
    return job_dir


def run_model(m: ModelRun) -> None:
    """Run one model and collect results."""
    print(f"\n{'='*60}")
    print(f"  Running {m.label} ({m.key})")
    print(f"{'='*60}")

    try:
        job_dir = prepare_job(m)
        builder = CMD_BUILDERS.get(m.key)
        if builder is None:
            m.status = "skipped"
            m.error = "No command builder"
            return
        cmd = builder(m, job_dir)
        print(f"  CMD: {cmd[:120]}...")

        t0 = time.time()
        result = ssh(cmd, timeout=600)
        m.elapsed = time.time() - t0

        # Print last 10 lines of output
        lines = (result.stdout or "").strip().split("\n")
        for line in lines[-10:]:
            print(f"  | {line}")
        if result.stderr:
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"  ! {line}")

        if result.returncode != 0:
            m.status = "failed"
            m.error = f"exit code {result.returncode}"
            return

        # Read scene_meta
        meta_result = ssh(f"cat {job_dir}/output/scene_meta.json 2>/dev/null", timeout=30)
        if meta_result.returncode == 0 and meta_result.stdout.strip():
            m.scene_meta = json.loads(meta_result.stdout)
            m.artifact_count = m.scene_meta.get("artifact_count", 0)

        # Read status.json
        status_result = ssh(f"cat {job_dir}/status.json 2>/dev/null", timeout=30)
        if status_result.returncode == 0:
            status_data = json.loads(status_result.stdout)
            if status_data.get("phase") == "finished":
                m.status = "success"
            else:
                m.status = status_data.get("phase", "unknown")
                m.error = status_data.get("message", "")
        else:
            m.status = "success"  # no status.json but exit 0

    except subprocess.TimeoutExpired:
        m.status = "timeout"
        m.error = "SSH command timed out (600s)"
    except Exception as exc:
        m.status = "error"
        m.error = str(exc)


def print_matrix(results: list[ModelRun]) -> None:
    """Print comparison matrix table."""
    print(f"\n{'='*80}")
    print("  CROSS-MODEL COMPARISON MATRIX")
    print(f"{'='*80}")
    print(f"  Sample: CUT3R examples/001 (13 frames, 1920x1080)")
    print(f"{'─'*80}")
    print(f"  {'Model':<12} {'Status':<12} {'Time(s)':<10} {'Artifacts':<12} {'Notes'}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*12} {'─'*30}")
    for m in results:
        notes = m.error[:30] if m.error else ""
        if m.scene_meta:
            groups = m.scene_meta.get("artifact_groups", [])
            notes = ", ".join(f"{g['key']}:{g['count']}" for g in groups if g.get("count", 0) > 0)
        print(f"  {m.label:<12} {m.status:<12} {m.elapsed:<10.1f} {m.artifact_count:<12} {notes}")
    print(f"{'='*80}")

    # Summary
    passed = sum(1 for m in results if m.status == "success")
    total = len(results)
    print(f"\n  Result: {passed}/{total} models passed on unified sample.")


def save_report(results: list[ModelRun], path: Path) -> None:
    """Save JSON report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample": "cut3r/examples/001 (13 frames 1920x1080)",
        "models": [],
    }
    for m in results:
        report["models"].append({
            "key": m.key,
            "label": m.label,
            "status": m.status,
            "elapsed_seconds": round(m.elapsed, 2),
            "artifact_count": m.artifact_count,
            "error": m.error,
            "artifact_groups": m.scene_meta.get("artifact_groups", []),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report saved to: {path}")


def main() -> None:
    print("Cross-model comparison smoke test")
    print("=" * 40)

    # 1. Upload runners
    print("\n[1/3] Uploading runners...")
    upload_runners()

    # 2. Run each model sequentially
    print("\n[2/3] Running models...")
    for m in MODELS:
        run_model(m)

    # 3. Print & save results
    print("\n[3/3] Results:")
    print_matrix(MODELS)
    report_path = ROOT / "scripts" / "cross_model_report.json"
    save_report(MODELS, report_path)


if __name__ == "__main__":
    main()
