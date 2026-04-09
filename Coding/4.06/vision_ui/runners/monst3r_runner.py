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


def main() -> None:
    parser = argparse.ArgumentParser(description="MonST3R preparation runner")
    parser.add_argument("--job-dir", required=True, help="Path to the remote job directory")
    parser.add_argument("--repo", required=True, help="Expected remote MonST3R repo path")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)

    write_status(job_dir, "starting", "正在检查 MonST3R 服务器环境...")

    repo = Path(args.repo)
    missing = []
    if not repo.exists():
        missing.append(f"缺少仓库目录：{repo}")
    elif not (repo / "demo.py").exists():
        missing.append(f"仓库目录存在，但未发现 demo.py：{repo}")

    message_lines = [
        "MonST3R 当前还处于部署准备阶段，前端骨架已经接好，但远端真实推理尚未完成集成。",
        "请先完成以下准备，再开始接视频/帧序列任务：",
        "1. 在服务器上拉取 MonST3R 仓库并确认 demo.py 可运行。",
        "2. 建立独立 conda 环境，并确认 torch / xformers / 依赖版本。",
        "3. 准备模型权重、示例视频或帧序列，并明确输出目录结构。",
        "4. 约定需要回传哪些产物，例如预览图、轨迹、点云、元数据和日志。",
    ]
    if missing:
        message_lines.append("")
        message_lines.append("当前检查到的缺失项：")
        message_lines.extend([f"- {item}" for item in missing])

    final_message = "\n".join(message_lines)
    write_status(job_dir, "failed", final_message)
    print(final_message, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
