from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from job_store import ROOT, get_job_dir, iter_input_items, load_job, update_job, write_result_summary


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@dataclass
class ServerConfig:
    alias: str = "KYKT-UI"
    host: str = "172.17.140.97"
    user: str = "kykt26"
    port: int = 22
    remote_root: str = "/hdd3/kykt26"
    remote_jobs_dir: str = "/hdd3/kykt26/jobs"
    remote_runners_dir: str = "/hdd3/kykt26/runners"
    remote_dust3r_repo: str = "/hdd3/kykt26/code/dust3r-main"
    remote_dust3r_model: str = "/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    remote_dust3r_env: str = "dust3r"
    remote_monst3r_repo: str = "/hdd3/kykt26/code/monst3r"
    remote_monst3r_env: str = "monst3r"


LOCAL_RUNNERS_DIR = ROOT / "runners"
SSH_CONNECT_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=12",
    "-o",
    "ServerAliveInterval=20",
    "-o",
    "ServerAliveCountMax=2",
]
SSH_SHORT_TIMEOUT_SECONDS = 120
SCP_TIMEOUT_SECONDS = 900
STATUS_POLL_INTERVAL_SECONDS = 4
STATUS_POLL_TIMEOUT_SECONDS = 20
REMOTE_PHASE_MAP = {
    "starting": "running_remote_matches",
    "running_matches": "running_remote_matches",
    "running_alignment": "running_remote_matches",
    "saving_outputs": "running_remote_matches",
    "exporting_pointcloud": "running_remote_pointcloud",
    "finished": "downloading_results",
    "failed": "failed",
}


def _ssh_command(config: ServerConfig, shell_script: str) -> list[str]:
    return ["ssh", *SSH_CONNECT_OPTIONS, config.alias, f"bash -lc {shlex.quote(shell_script)}"]


def run_remote_job(job_id: str) -> None:
    job = load_job(job_id)
    config = ServerConfig()

    try:
        remote_job_dir = f"{config.remote_jobs_dir}/{job.job_id}"
        update_job(
            job_id,
            status="running",
            phase="preparing_remote",
            remote_job_dir=remote_job_dir,
            progress_message="正在创建远端任务目录...",
        )

        # Create remote directories
        _ssh(
            config,
            (
                f"mkdir -p {shlex.quote(remote_job_dir)}/input "
                f"{shlex.quote(remote_job_dir)}/output "
                f"{shlex.quote(remote_job_dir)}/logs"
            ),
        )

        # Ensure runners dir exists
        _ssh(config, f"mkdir -p {shlex.quote(config.remote_runners_dir)}")

        # Upload inputs
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传本地输入文件到服务器...")
        _upload_inputs(config, job.job_id, remote_job_dir)

        # Upload job.json
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传任务清单...")
        _upload_remote_job_json(config, job.job_id, remote_job_dir)

        # Upload runner script
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传远端运行脚本...")
        _upload_runner(config, job.model)

        # Dispatch model
        if job.model == "dust3r":
            _run_dust3r_v2(config, job.job_id, remote_job_dir)
        elif job.model == "monst3r":
            _run_monst3r_v1(config, job.job_id, remote_job_dir)
        else:
            raise RuntimeError(f"模型 '{job.model}' 还没有接入远端执行。")

        # Download results
        update_job(
            job_id,
            phase="downloading_results",
            progress_message="正在把输出和日志下载回本地缓存...",
        )
        output_files = _download_results(config, job.job_id, remote_job_dir)
        if load_job(job_id).status == "cancelled":
            return
        _generate_result_summary(job_id, output_files)
        update_job(
            job_id,
            status="finished",
            phase="finished",
            output_files=output_files,
            error_message=None,
            progress_message="任务完成。输出结果已回传到本地。",
        )
    except Exception as exc:
        if load_job(job_id).status == "cancelled":
            return
        update_job(
            job_id,
            status="failed",
            phase="failed",
            error_message=str(exc),
            progress_message="远端任务失败，请查看下方日志。",
        )


def cancel_remote_job(job_id: str) -> None:
    config = ServerConfig()
    job = load_job(job_id)
    remote_job_dir = job.remote_job_dir

    cleanup_message = "已请求取消任务。"
    if remote_job_dir:
        try:
            _ssh(config, f"pkill -f {shlex.quote(remote_job_dir)} || true")
            cleanup_message = "已请求取消任务，并尝试清理远端进程。"
        except Exception as exc:
            cleanup_message = f"已在本地取消任务，但远端清理失败：{exc}"

    update_job(
        job_id,
        status="cancelled",
        phase="cancelled",
        error_message=None,
        progress_message=cleanup_message,
    )


def _upload_inputs(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    job = load_job(job_id)
    items = iter_input_items(job)
    total = len(items)
    for idx, item in enumerate(items, start=1):
        local_path = ROOT / item["relative_path"]
        update_job(
            job_id,
            progress_message=f"正在上传输入 {idx}/{total}: {item['stored_name']}",
        )
        _scp_to_remote(config, local_path, f"{remote_job_dir}/input/{item['stored_name']}")


def _upload_remote_job_json(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    job = load_job(job_id)
    remote_payload = job.to_dict()
    remote_payload["remote_job_dir"] = remote_job_dir
    remote_payload["remote_input_dir"] = f"{remote_job_dir}/input"
    remote_payload["remote_output_dir"] = f"{remote_job_dir}/output"
    remote_payload["remote_logs_dir"] = f"{remote_job_dir}/logs"

    local_tmp = get_job_dir(job_id) / "remote_job.json"
    local_tmp.write_text(json.dumps(remote_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _scp_to_remote(config, local_tmp, f"{remote_job_dir}/job.json")


def _upload_runner(config: ServerConfig, model: str) -> None:
    runner_map = {
        "dust3r": "dust3r_runner.py",
        "monst3r": "monst3r_runner.py",
    }
    runner_file = runner_map.get(model)
    if not runner_file:
        return

    local_runner = LOCAL_RUNNERS_DIR / runner_file
    if local_runner.exists():
        _scp_to_remote(config, local_runner, f"{config.remote_runners_dir}/{runner_file}")


def _run_dust3r_v2(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    """Run DUSt3R using the unified server-side runner (supports N images)."""
    job = load_job(job_id)
    input_items = iter_input_items(job)
    n_images = len(input_items)
    params = job.params or {}

    if n_images < 2:
        raise RuntimeError("DUSt3R 至少需要两张已上传图片。")

    runner_path = f"{config.remote_runners_dir}/dust3r_runner.py"
    log_path = f"{remote_job_dir}/logs/runner.log"
    local_log = get_job_dir(job_id) / "logs" / "runner.live.log"

    cmd = (
        f"set -o pipefail && "
        f"cd {shlex.quote(config.remote_dust3r_repo)} && "
        f"conda run -n {shlex.quote(config.remote_dust3r_env)} "
        f"python {shlex.quote(runner_path)} "
        f"--job-dir {shlex.quote(remote_job_dir)} "
        f"--model {shlex.quote(config.remote_dust3r_model)} "
        f"--repo {shlex.quote(config.remote_dust3r_repo)} "
        f"--image-size {shlex.quote(str(params.get('image_size', 512)))} "
        f"--scene-graph {shlex.quote(str(params.get('scene_graph', 'complete')))} "
        f"--niter {shlex.quote(str(params.get('niter', 300)))} "
        f"--lr {shlex.quote(str(params.get('lr', 0.01)))} "
        f"--batch-size {shlex.quote(str(params.get('batch_size', 1)))} "
        f"--max-points {shlex.quote(str(params.get('max_points', 250000)))} "
        f"--match-viz-count {shlex.quote(str(params.get('match_viz_count', 50)))} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )

    update_job(
        job_id,
        phase="running_remote_matches",
        progress_message=f"正在使用 {n_images} 张图片启动 DUSt3R...",
    )
    _ssh_stream(
        config,
        cmd,
        job_id=job_id,
        phase="running_remote_matches",
        remote_job_dir=remote_job_dir,
        local_log_path=local_log,
    )


def _run_monst3r_v1(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    runner_path = f"{config.remote_runners_dir}/monst3r_runner.py"
    log_path = f"{remote_job_dir}/logs/runner.log"
    local_log = get_job_dir(job_id) / "logs" / "runner.live.log"

    cmd = (
        f"set -o pipefail && "
        f"cd {shlex.quote(config.remote_monst3r_repo)} 2>/dev/null || cd {shlex.quote(remote_job_dir)} && "
        f"python {shlex.quote(runner_path)} "
        f"--job-dir {shlex.quote(remote_job_dir)} "
        f"--repo {shlex.quote(config.remote_monst3r_repo)} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )

    update_job(
        job_id,
        phase="running_remote_matches",
        progress_message="正在执行 MonST3R 远端准备检查...",
    )
    _ssh_stream(
        config,
        cmd,
        job_id=job_id,
        phase="running_remote_matches",
        remote_job_dir=remote_job_dir,
        local_log_path=local_log,
    )


def _download_results(config: ServerConfig, job_id: str, remote_job_dir: str) -> list[str]:
    job_dir = get_job_dir(job_id)
    local_output_dir = job_dir / "output"
    local_logs_dir = job_dir / "logs"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_logs_dir.mkdir(parents=True, exist_ok=True)

    required_downloads = [
        ("output/matches.png", local_output_dir / "matches.png"),
        ("output/pointcloud.ply", local_output_dir / "pointcloud.ply"),
    ]
    optional_downloads = [
        ("logs/runner.log", local_logs_dir / "runner.log"),
        ("output/scene_meta.json", local_output_dir / "scene_meta.json"),
    ]

    output_files: list[str] = []
    for remote_suffix, local_path in required_downloads:
        remote_path = f"{remote_job_dir}/{remote_suffix}"
        _scp_from_remote(config, remote_path, local_path)
        output_files.append(str(local_path.relative_to(ROOT)))

    for remote_suffix, local_path in optional_downloads:
        remote_path = f"{remote_job_dir}/{remote_suffix}"
        try:
            _scp_from_remote(config, remote_path, local_path)
            output_files.append(str(local_path.relative_to(ROOT)))
        except subprocess.CalledProcessError:
            pass

    return output_files


def _generate_result_summary(job_id: str, output_files: list[str]) -> None:
    job = load_job(job_id)
    job_dir = get_job_dir(job_id)
    scene_meta_path = job_dir / "output" / "scene_meta.json"
    scene_meta = None
    if scene_meta_path.exists():
        try:
            scene_meta = json.loads(scene_meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            scene_meta = None

    created_at = None
    try:
        created_at = datetime.fromisoformat(job.created_at)
    except ValueError:
        created_at = None
    duration_seconds = None
    if created_at is not None:
        duration_seconds = max(0, int((datetime.now() - created_at).total_seconds()))

    highlights = [
        f"本次任务共处理 {len(job.input_files)} 个输入文件。",
        f"共回传 {len(output_files)} 个本地产物。",
    ]
    if scene_meta:
        if scene_meta.get("n_pairs") is not None:
            highlights.append(f"远端共构建了 {scene_meta['n_pairs']} 个图像配对。")
        if scene_meta.get("n_points") is not None:
            highlights.append(f"最终导出的点云包含 {scene_meta['n_points']} 个点。")
        if scene_meta.get("raw_point_count") is not None and scene_meta.get("n_points") is not None:
            raw_points = scene_meta["raw_point_count"]
            final_points = scene_meta["n_points"]
            if raw_points != final_points:
                highlights.append(f"点云从 {raw_points} 个原始点下采样到了 {final_points} 个点。")

    next_actions = [
        "优先在 MeshLab 中检查 pointcloud.ply 的结构是否完整、是否存在大块噪声或断裂。",
        "结合 matches.png 判断前几张图的重叠区域和匹配是否合理。",
    ]
    if job.model == "dust3r":
        next_actions.append("如果这是多图任务，建议再对比 scene graph 与点云质量，决定是否需要改成 swin-5 或调整点云上限。")
    elif job.model == "monst3r":
        next_actions.append("当前 MonST3R 仍处于部署准备阶段，先完成服务器环境和权重检查，再尝试真实视频任务。")

    payload = {
        "job_id": job.job_id,
        "model": job.model,
        "status": job.status,
        "status_label": "已完成" if job.status == "finished" else job.status,
        "source_type": job.source_type,
        "created_at": job.created_at,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "inputs": {
            "count": len(job.input_files),
            "names": [item["original_name"] for item in iter_input_items(job)],
        },
        "artifacts": [
            {
                "name": Path(rel_path).name,
                "relative_path": rel_path,
            }
            for rel_path in output_files
        ],
        "params": job.params,
        "scene_meta": scene_meta,
        "highlights": highlights,
        "next_actions": next_actions,
    }
    write_result_summary(job_id, payload)


# ===== Low-level SSH/SCP helpers =====

def _ssh(config: ServerConfig, shell_script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        _ssh_command(config, shell_script),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SSH_SHORT_TIMEOUT_SECONDS,
    )


def _clean_progress_line(raw_line: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", raw_line)
    cleaned = cleaned.replace("\r", "\n").replace("\x00", "")
    cleaned = "\n".join(segment.strip() for segment in cleaned.splitlines() if segment.strip())
    return cleaned.strip()


def _ssh_stream(
    config: ServerConfig,
    shell_script: str,
    *,
    job_id: str,
    phase: str,
    remote_job_dir: str,
    local_log_path: Path,
) -> None:
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    remote_cmd = _ssh_command(config, shell_script)
    process = subprocess.Popen(
        remote_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    last_message = ""
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poll_remote_status,
        args=(config, job_id, remote_job_dir, stop_event),
        daemon=True,
    )
    poller.start()

    try:
        with local_log_path.open("w", encoding="utf-8") as log_file:
            assert process.stdout is not None
            for raw_line in process.stdout:
                cleaned = _clean_progress_line(raw_line)
                log_file.write(cleaned + ("\n" if cleaned else ""))
                log_file.flush()
                if cleaned:
                    last_message = cleaned[-400:]
                    # Auto-detect phase transitions from runner output
                    detected_phase = phase
                    if "alignment" in cleaned.lower():
                        detected_phase = "running_remote_matches"
                    elif "point cloud" in cleaned.lower() or "exporting" in cleaned.lower():
                        detected_phase = "running_remote_pointcloud"
                    update_job(job_id, phase=detected_phase, progress_message=last_message)

        return_code = process.wait()
    finally:
        stop_event.set()
        poller.join(timeout=2)
        _sync_remote_status_once(config, job_id, remote_job_dir)

    if return_code != 0:
        raise RuntimeError(
            f"远端命令在阶段 {phase} 失败。"
            f"最后一条日志：{last_message or '没有捕获到远端日志。'}"
        )


def _poll_remote_status(config: ServerConfig, job_id: str, remote_job_dir: str, stop_event: threading.Event) -> None:
    while not stop_event.wait(STATUS_POLL_INTERVAL_SECONDS):
        _sync_remote_status_once(config, job_id, remote_job_dir)


def _sync_remote_status_once(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    local_status = get_job_dir(job_id) / "logs" / "remote_status.json"
    remote_status = f"{remote_job_dir}/status.json"
    try:
        _scp_from_remote(config, remote_status, local_status, timeout=STATUS_POLL_TIMEOUT_SECONDS)
        payload = json.loads(local_status.read_text(encoding="utf-8-sig"))
    except Exception:
        return

    remote_phase = str(payload.get("phase") or "").strip()
    local_phase = REMOTE_PHASE_MAP.get(remote_phase, "running_remote_matches")
    message = str(payload.get("message") or payload.get("progress") or remote_phase or "远端状态已更新。")
    progress = str(payload.get("progress") or "").strip()
    progress_message = f"{message} ({progress})" if progress else message

    if load_job(job_id).status != "running":
        return

    status = "failed" if remote_phase == "failed" else None
    update_job(job_id, status=status, phase=local_phase, progress_message=progress_message)


def _scp_to_remote(config: ServerConfig, local_path: Path, remote_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["scp", *SSH_CONNECT_OPTIONS, str(local_path), f"{config.alias}:{remote_path}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SCP_TIMEOUT_SECONDS,
    )


def _scp_from_remote(
    config: ServerConfig,
    remote_path: str,
    local_path: Path,
    *,
    timeout: int = SCP_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["scp", *SSH_CONNECT_OPTIONS, f"{config.alias}:{remote_path}", str(local_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
