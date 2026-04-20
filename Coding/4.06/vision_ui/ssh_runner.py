from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tarfile
import threading
import traceback
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
    remote_mast3r_repo: str = "/hdd3/kykt26/code/mast3r"
    remote_mast3r_model: str = "/hdd3/kykt26/code/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    remote_mast3r_env: str = "mast3r"
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
WINDOWS_NO_WINDOW = 0x08000000
REMOTE_PHASE_MAP = {
    "starting": "running_remote_matches",
    "running_matches": "running_remote_matches",
    "running_alignment": "running_remote_matches",
    "saving_outputs": "running_remote_pointcloud",
    "exporting_pointcloud": "running_remote_pointcloud",
    "finished": "downloading_results",
    "failed": "failed",
}


def _ssh_command(config: ServerConfig, shell_script: str) -> list[str]:
    return ["ssh", "-T", *SSH_CONNECT_OPTIONS, config.alias, f"bash -lc {shlex.quote(shell_script)}"]


def _subprocess_options() -> dict:
    options: dict = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        options["creationflags"] = WINDOWS_NO_WINDOW
    return options


def _write_debug_log(job_id: str, message: str) -> None:
    log_path = get_job_dir(job_id) / "logs" / "dispatch.debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _raise_if_cancelled(job_id: str) -> None:
    if load_job(job_id).status == "cancelled":
        raise RuntimeError("__job_cancelled__")


def run_remote_job(job_id: str) -> None:
    job = load_job(job_id)
    config = ServerConfig()

    try:
        _write_debug_log(job_id, "Remote job thread started.")
        remote_job_dir = f"{config.remote_jobs_dir}/{job.job_id}"
        update_job(
            job_id,
            status="running",
            phase="preparing_remote",
            remote_job_dir=remote_job_dir,
            progress_message="正在创建远端任务目录...",
        )
        _write_debug_log(job_id, f"Preparing remote directory: {remote_job_dir}")

        # Create remote directories
        _raise_if_cancelled(job_id)
        _ssh(
            config,
            (
                f"mkdir -p {shlex.quote(remote_job_dir)}/input "
                f"{shlex.quote(remote_job_dir)}/output "
                f"{shlex.quote(remote_job_dir)}/logs"
            ),
        )
        _write_debug_log(job_id, "Remote job directories created.")

        # Ensure runners dir exists
        _raise_if_cancelled(job_id)
        update_job(job_id, phase="preparing_remote", progress_message="正在确认远端运行脚本目录...")
        _ssh(config, f"mkdir -p {shlex.quote(config.remote_runners_dir)}")
        _write_debug_log(job_id, f"Remote runners dir ready: {config.remote_runners_dir}")

        # Upload inputs
        _raise_if_cancelled(job_id)
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传本地输入文件到服务器...")
        _write_debug_log(job_id, "Uploading inputs...")
        _upload_inputs(config, job.job_id, remote_job_dir)
        _write_debug_log(job_id, "Inputs uploaded.")

        # Upload job.json
        _raise_if_cancelled(job_id)
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传任务清单...")
        _upload_remote_job_json(config, job.job_id, remote_job_dir)
        _write_debug_log(job_id, "Remote job.json uploaded.")

        # Upload runner script
        _raise_if_cancelled(job_id)
        update_job(job_id, phase="uploading_inputs", progress_message="正在上传远端运行脚本...")
        _upload_runner(config, job.model)
        _write_debug_log(job_id, f"Remote runner prepared for model: {job.model}")

        # Dispatch model
        _raise_if_cancelled(job_id)
        if job.model == "dust3r":
            _run_dust3r_v2(config, job.job_id, remote_job_dir)
        elif job.model == "mast3r":
            _run_mast3r_v1(config, job.job_id, remote_job_dir)
        elif job.model == "monst3r":
            _run_monst3r_v1(config, job.job_id, remote_job_dir)
        else:
            raise RuntimeError(f"模型 '{job.model}' 还没有接入远端执行。")
        _write_debug_log(job_id, "Remote model execution finished, starting download.")

        # Download results
        _raise_if_cancelled(job_id)
        update_job(
            job_id,
            phase="downloading_results",
            progress_message="正在把输出和日志下载回本地缓存...",
        )
        output_files = _download_results(config, job.job_id, remote_job_dir)
        if load_job(job_id).status == "cancelled":
            _write_debug_log(job_id, "Job was cancelled during result download.")
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
        _write_debug_log(job_id, f"Job finished successfully with {len(output_files)} files.")
    except Exception as exc:
        if str(exc) == "__job_cancelled__":
            _write_debug_log(job_id, "Job cancellation acknowledged; stopping remote workflow.")
            return
        _write_debug_log(job_id, "Job failed with exception:\n" + traceback.format_exc())
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
            _kill_remote_job_processes(config, remote_job_dir)
            cleanup_message = "已请求取消任务，并尝试清理远端进程。"
        except Exception as exc:
            cleanup_message = f"已在本地取消任务；远端清理未确认成功，可稍后检查 GPU 进程。原因：{exc}"

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
        _raise_if_cancelled(job_id)
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
        "mast3r": "mast3r_runner.py",
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
        f"conda run --no-capture-output -n {shlex.quote(config.remote_dust3r_env)} "
        f"python -u {shlex.quote(runner_path)} "
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


def _run_mast3r_v1(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    """Run MASt3R using the DUSt3R-compatible global alignment flow."""
    job = load_job(job_id)
    input_items = iter_input_items(job)
    n_images = len(input_items)
    params = job.params or {}

    if n_images < 2:
        raise RuntimeError("MASt3R 至少需要两张已上传图片。")

    runner_path = f"{config.remote_runners_dir}/mast3r_runner.py"
    log_path = f"{remote_job_dir}/logs/runner.log"
    local_log = get_job_dir(job_id) / "logs" / "runner.live.log"

    cmd = (
        f"set -o pipefail && "
        f"cd {shlex.quote(config.remote_mast3r_repo)} && "
        f"conda run --no-capture-output -n {shlex.quote(config.remote_mast3r_env)} "
        f"python -u {shlex.quote(runner_path)} "
        f"--job-dir {shlex.quote(remote_job_dir)} "
        f"--model {shlex.quote(config.remote_mast3r_model)} "
        f"--repo {shlex.quote(config.remote_mast3r_repo)} "
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
        progress_message=f"正在使用 {n_images} 张图片启动 MASt3R...",
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
    weights_path = f"{config.remote_monst3r_repo}/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"

    cmd = (
        f"set -o pipefail && "
        f"cd {shlex.quote(config.remote_monst3r_repo)} && "
        f"conda run --no-capture-output -n {shlex.quote(config.remote_monst3r_env)} "
        f"python -u {shlex.quote(runner_path)} "
        f"--job-dir {shlex.quote(remote_job_dir)} "
        f"--repo {shlex.quote(config.remote_monst3r_repo)} "
        f"--weights {shlex.quote(weights_path)} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )

    update_job(
        job_id,
        phase="running_remote_matches",
        progress_message="正在启动 MonST3R 官方 demo 推理...",
    )
    _ssh_stream(
        config,
        cmd,
        job_id=job_id,
        phase="running_remote_matches",
        remote_job_dir=remote_job_dir,
        local_log_path=local_log,
    )


def _kill_remote_job_processes(config: ServerConfig, remote_job_dir: str) -> None:
    """Terminate only MonST3R/DUSt3R processes that reference this job directory.

    Avoid `pkill -f <job_dir>` because that can match and kill the SSH shell
    currently running the cancellation command, which reports as a scary local
    SSH failure even when the user's intent was simply "cancel".
    """
    script = f"""
python3 - <<'PY'
import os
import signal
import subprocess
import time

job = {remote_job_dir!r}
needles = ("monst3r_runner.py", "dust3r_runner.py", "mast3r_runner.py", "demo.py")
current = os.getpid()

def matching_pids():
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, errors="ignore")
    pids = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current:
            continue
        if job in args and any(needle in args for needle in needles):
            pids.append(pid)
    return sorted(set(pids))

pids = matching_pids()
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

time.sleep(2)
for pid in matching_pids():
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

print("cancelled_pids=" + ",".join(map(str, pids)))
PY
"""
    _ssh(config, script)


def _download_results(config: ServerConfig, job_id: str, remote_job_dir: str) -> list[str]:
    job = load_job(job_id)
    job_dir = get_job_dir(job_id)
    local_output_dir = job_dir / "output"
    local_logs_dir = job_dir / "logs"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_logs_dir.mkdir(parents=True, exist_ok=True)

    if job.model == "monst3r":
        return _download_remote_tree(config, remote_job_dir, job_dir)

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


def _download_remote_tree(config: ServerConfig, remote_job_dir: str, local_job_dir: Path) -> list[str]:
    remote_archive = f"{remote_job_dir}/result_bundle.tar.gz"
    local_archive = local_job_dir / "logs" / "remote_results.tar.gz"
    _ssh(
        config,
        f"cd {shlex.quote(remote_job_dir)} && tar -czf {shlex.quote(remote_archive)} output logs",
    )
    _scp_from_remote(config, remote_archive, local_archive)
    _safe_extract_tar(local_archive, local_job_dir)

    output_files = []
    for folder_name in ("output", "logs"):
        folder = local_job_dir / folder_name
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path == local_archive:
                continue
            output_files.append(str(path.relative_to(ROOT)))

    if not any(path.replace("\\", "/").endswith("output/scene_meta.json") for path in output_files):
        raise RuntimeError("MonST3R 远端执行结束，但没有下载到 output/scene_meta.json。")

    return output_files


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (target_root / member.name).resolve()
            try:
                member_path.relative_to(target_root)
            except ValueError as exc:
                raise RuntimeError(f"远端结果压缩包包含非法路径：{member.name}") from exc
        archive.extractall(target_root)


MONST3R_ARTIFACT_ROLE_LABELS = {
    "scene": ("三维场景", "优先打开检查主体结构、相机轨迹和动态区域。"),
    "trajectory": ("相机轨迹", "用于判断相机运动是否连续、是否出现明显漂移。"),
    "intrinsics": ("相机内参", "用于复查焦距和相机参数是否成功导出。"),
    "frame_preview": ("彩色帧预览", "用于快速确认抽帧质量、曝光和运动模糊。"),
    "dynamic_mask": ("动态区域", "用于判断运动物体或动态区域是否被识别。"),
    "confidence": ("置信数组", "用于后续诊断深度/几何估计稳定性。"),
    "initial_confidence": ("初始置信数组", "MonST3R 中间置信产物。"),
    "geometry_array": ("几何数组", "每帧对应的几何/深度数组。"),
    "array": ("其他数组", "其他 NPY 中间产物。"),
    "image": ("其他图像", "其他可视化图像。"),
    "other": ("其他产物", "未归入主检查路径的产物。"),
}


def _monst3r_artifact_role(filename: str) -> str:
    lower = filename.lower()
    suffix = Path(lower).suffix
    if lower.endswith(".glb") or lower.endswith(".gltf"):
        return "scene"
    if lower == "pred_traj.txt" or "traj" in lower:
        return "trajectory"
    if lower == "pred_intrinsics.txt" or "intrinsics" in lower:
        return "intrinsics"
    if lower.startswith("frame_") and suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return "frame_preview"
    if "dynamic_mask" in lower and suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return "dynamic_mask"
    if lower.startswith("conf_") and suffix == ".npy":
        return "confidence"
    if lower.startswith("init_conf_") and suffix == ".npy":
        return "initial_confidence"
    if lower.startswith("frame_") and suffix == ".npy":
        return "geometry_array"
    if suffix == ".npy":
        return "array"
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return "image"
    return "other"


def _summarize_monst3r_outputs(output_files: list[str], scene_meta: dict | None) -> tuple[list[dict], list[dict]]:
    if scene_meta and isinstance(scene_meta.get("artifact_groups"), list):
        groups = scene_meta["artifact_groups"]
    else:
        counts: dict[str, int] = {}
        for rel_path in output_files:
            if Path(rel_path).suffix.lower() in {".json", ".log"}:
                continue
            role = _monst3r_artifact_role(Path(rel_path).name)
            counts[role] = counts.get(role, 0) + 1
        groups = [
            {
                "key": key,
                "label": label,
                "count": counts[key],
                "description": description,
            }
            for key, (label, description) in MONST3R_ARTIFACT_ROLE_LABELS.items()
            if counts.get(key)
        ]

    if scene_meta and isinstance(scene_meta.get("review_targets"), list):
        targets = scene_meta["review_targets"]
    else:
        records = []
        for rel_path in output_files:
            if Path(rel_path).suffix.lower() in {".json", ".log"}:
                continue
            name = Path(rel_path).name
            role = _monst3r_artifact_role(name)
            label, note = MONST3R_ARTIFACT_ROLE_LABELS.get(role, MONST3R_ARTIFACT_ROLE_LABELS["other"])
            records.append({"role": role, "label": label, "name": name, "relative_path": rel_path, "note": note})

        targets = []
        for role in ("scene", "trajectory", "intrinsics"):
            targets.extend([item for item in records if item["role"] == role][:1])
        frames = [item for item in records if item["role"] == "frame_preview"]
        if len(frames) > 3:
            indexes = sorted({0, len(frames) // 2, len(frames) - 1})
            targets.extend(frames[index] for index in indexes)
        else:
            targets.extend(frames)

    return groups, targets


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
    artifact_groups: list[dict] = []
    primary_artifacts: list[dict] = []
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
        if scene_meta.get("artifact_count") is not None:
            highlights.append(f"MonST3R 官方 demo 共整理了 {scene_meta['artifact_count']} 个远端产物。")
        if scene_meta.get("glb_count") is not None:
            highlights.append(f"其中包含 {scene_meta['glb_count']} 个 GLB 三维场景文件。")

    if job.model == "monst3r":
        artifact_groups, primary_artifacts = _summarize_monst3r_outputs(output_files, scene_meta)
        group_counts = {item.get("key"): item.get("count", 0) for item in artifact_groups}
        if group_counts.get("frame_preview"):
            highlights.append(f"已生成 {group_counts['frame_preview']} 张彩色帧预览，可用于快速检查抽帧质量。")
        if group_counts.get("dynamic_mask"):
            highlights.append(f"已生成 {group_counts['dynamic_mask']} 张动态区域 mask，可辅助判断运动物体影响。")
        if group_counts.get("confidence") or group_counts.get("initial_confidence"):
            conf_count = int(group_counts.get("confidence") or 0) + int(group_counts.get("initial_confidence") or 0)
            highlights.append(f"已生成 {conf_count} 个置信数组，可用于后续质量诊断。")

    if job.model in {"dust3r", "mast3r"}:
        next_actions = [
            "优先在 MeshLab 中检查 pointcloud.ply 的结构是否完整、是否存在大块噪声或断裂。",
            "结合 matches.png 判断前几张图的重叠区域和匹配是否合理。",
        ]
        next_actions.append("如果这是多图任务，建议再对比 scene graph 与点云质量，决定是否需要改成 swin-5 或调整点云上限。")
        if job.model == "mast3r":
            next_actions.insert(0, "MASt3R 更偏静态多图匹配增强，建议优先拿同一物体的 3 到 8 张图验证它相对 DUSt3R 的提升。")
    elif job.model == "monst3r":
        next_actions = [
            "按核心检查对象依次看 scene.glb、pred_traj.txt、pred_intrinsics.txt 和代表帧预览。",
            "对照动态 mask 和置信数组，判断运动区域、深度稳定性和视频输入是否适合作为展示样例。",
            "如果 GLB 结构差，优先换更稳定的视频输入；如果只是耗时或显存压力大，再降低 Image Size 或 Num Frames。",
        ]
    else:
        next_actions = ["查看输出产物和 runner.log，确认模型任务是否符合预期。"]

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
        "artifact_groups": artifact_groups,
        "primary_artifacts": primary_artifacts,
        "highlights": highlights,
        "next_actions": next_actions,
    }
    write_result_summary(job_id, payload)


# ===== Low-level SSH/SCP helpers =====

def _ssh(config: ServerConfig, shell_script: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            _ssh_command(config, shell_script),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SSH_SHORT_TIMEOUT_SECONDS,
            **_subprocess_options(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH 命令超时（{SSH_SHORT_TIMEOUT_SECONDS}s）：{shell_script[:160]}") from exc


def _clean_progress_line(raw_line: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", raw_line)
    cleaned = cleaned.replace("\r", "\n").replace("\x00", "")
    cleaned = "\n".join(segment.strip() for segment in cleaned.splitlines() if segment.strip())
    return cleaned.strip()


def _displayable_remote_progress(cleaned: str) -> str | None:
    """Return a user-facing progress line, filtering noisy framework warnings."""
    if not cleaned:
        return None

    lower = cleaned.lower()
    ignored_fragments = [
        "futurewarning",
        "torch.cuda.amp.autocast",
        "cannot find cuda-compiled version of rope2d",
        "using a slow pytorch version instead",
    ]
    if any(fragment in lower for fragment in ignored_fragments):
        return None

    if cleaned.startswith("MonST3R command:"):
        return "MonST3R 命令已启动，正在等待模型加载和推理输出..."
    if cleaned.startswith("DUSt3R command:"):
        return "DUSt3R 命令已启动，正在等待模型加载和重建输出..."
    return cleaned


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
        **_subprocess_options(),
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
                display_line = _displayable_remote_progress(cleaned)
                if display_line:
                    last_message = display_line[-400:]
                    # Auto-detect phase transitions from runner output
                    detected_phase = phase
                    lowered = display_line.lower()
                    if "alignment" in lowered:
                        detected_phase = "running_remote_matches"
                    elif "point cloud" in lowered or "exporting" in lowered:
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
    try:
        return subprocess.run(
            ["scp", *SSH_CONNECT_OPTIONS, str(local_path), f"{config.alias}:{remote_path}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SCP_TIMEOUT_SECONDS,
            **_subprocess_options(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SCP 上传超时（{SCP_TIMEOUT_SECONDS}s）：{local_path.name}") from exc


def _scp_from_remote(
    config: ServerConfig,
    remote_path: str,
    local_path: Path,
    *,
    timeout: int = SCP_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.run(
            ["scp", *SSH_CONNECT_OPTIONS, f"{config.alias}:{remote_path}", str(local_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **_subprocess_options(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SCP 下载超时（{timeout}s）：{remote_path}") from exc
