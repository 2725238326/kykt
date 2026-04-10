#!/usr/bin/env python3
"""
MonST3R server-side preparation runner.

This is intentionally a deployment-preparation skeleton. It does not yet run
full MonST3R inference. Instead, it validates the remote layout and writes a
clear status file so the local frontend can surface an actionable message.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def write_status(job_dir: Path, phase: str, message: str, progress: str = "") -> None:
    payload = {
        "phase": phase,
        "message": message,
        "progress": progress,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (job_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_check(command: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or "").strip()
    return completed.returncode == 0, output


def main() -> None:
    parser = argparse.ArgumentParser(description="MonST3R preparation runner")
    parser.add_argument("--job-dir", required=True, help="Path to the remote job directory")
    parser.add_argument("--repo", required=True, help="Expected remote MonST3R repo path")
    parser.add_argument("--env", default="monst3r", help="Expected conda env name")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)

    write_status(job_dir, "starting", "正在检查 MonST3R 服务器环境...")

    repo = Path(args.repo)
    missing = []
    checks = []

    if not repo.exists():
        missing.append(f"缺少仓库目录：{repo}")
    elif not (repo / "demo.py").exists():
        missing.append(f"仓库目录存在，但未发现 demo.py：{repo}")
    else:
        checks.append(f"仓库目录正常：{repo}")

    write_status(job_dir, "starting", "正在检查 MonST3R conda 环境...")
    env_ok, env_output = run_check(["conda", "run", "-n", args.env, "python", "-c", "import torch; print(torch.__version__)"])
    if env_ok:
        checks.append(f"conda 环境可用：{args.env}（torch={env_output.splitlines()[-1] if env_output else 'unknown'}）")
    else:
        missing.append(f"conda 环境不可用：{args.env}")

    write_status(job_dir, "starting", "正在检查 MonST3R 所需权重...")
    required_paths = [
        repo / "checkpoints" / "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth",
        repo / "third_party" / "RAFT" / "models" / "Tartan-C-T-TSKH-spring540x960-M.pth",
        repo / "third_party" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt",
    ]
    for path in required_paths:
        if path.exists():
            checks.append(f"权重已就位：{path}")
        else:
            missing.append(f"缺少权重：{path}")

    if repo.exists() and (repo / "demo.py").exists() and env_ok:
        write_status(job_dir, "starting", "正在执行 MonST3R demo.py 冒烟检查...")
        help_ok, help_output = run_check(["conda", "run", "-n", args.env, "python", "demo.py", "--help"], cwd=repo)
        if help_ok:
            checks.append("demo.py --help 可正常运行，说明基础依赖已基本打通。")
        else:
            missing.append("demo.py --help 失败，说明依赖仍有缺口。")
            if help_output:
                checks.append(f"demo.py --help 输出摘要：{help_output.splitlines()[-1]}")

    report_payload = {
        "repo": str(repo),
        "env": args.env,
        "checks": checks,
        "missing": missing,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (job_dir / "output" / "preparation_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    message_lines = [
        "MonST3R 当前仍按“部署检查模式”运行：前端入口已经接好，但正式推理 runner 还未完成最后集成。",
    ]
    if checks:
        message_lines.append("")
        message_lines.append("已通过的检查：")
        message_lines.extend([f"- {item}" for item in checks])

    if missing:
        message_lines.append("")
        message_lines.append("当前检查到的缺失项：")
        message_lines.extend([f"- {item}" for item in missing])
        message_lines.append("")
        message_lines.append("建议下一步：")
        message_lines.append("1. 用本地下载 + scp 或 Electerm SFTP 把缺失权重传到服务器。")
        message_lines.append("2. 在服务器终端运行 `python demo.py --help` 做最后一次环境冒烟检查。")
        message_lines.append("3. 再用一段视频或一组帧序列跑官方 demo，确认标准输出目录结构。")
    else:
        message_lines.append("")
        message_lines.append("环境与权重检查已基本通过，可以开始跑官方 demo，并把真实推理接入前端。")

    final_message = "\n".join(message_lines)
    write_status(job_dir, "failed", final_message)
    print(final_message, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
