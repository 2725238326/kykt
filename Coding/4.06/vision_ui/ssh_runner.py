from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from job_store import ROOT, get_job_dir, iter_input_items, load_job, update_job


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
            progress_message="Creating the remote job directory...",
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
        update_job(job_id, phase="uploading_inputs", progress_message="Uploading local inputs to the server...")
        _upload_inputs(config, job.job_id, remote_job_dir)

        # Upload job.json
        update_job(job_id, phase="uploading_inputs", progress_message="Uploading job manifest...")
        _upload_remote_job_json(config, job.job_id, remote_job_dir)

        # Upload runner script
        update_job(job_id, phase="uploading_inputs", progress_message="Uploading runner script...")
        _upload_runner(config, job.model)

        # Dispatch model
        if job.model == "dust3r":
            _run_dust3r_v2(config, job.job_id, remote_job_dir)
        else:
            raise RuntimeError(f"Model '{job.model}' is not wired yet.")

        # Download results
        update_job(
            job_id,
            phase="downloading_results",
            progress_message="Downloading outputs and logs back to the local cache...",
        )
        output_files = _download_results(config, job.job_id, remote_job_dir)
        update_job(
            job_id,
            status="finished",
            phase="finished",
            output_files=output_files,
            error_message=None,
            progress_message="Finished. Outputs are available below.",
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            phase="failed",
            error_message=str(exc),
            progress_message="Remote job failed. Check the live logs below.",
        )


def _upload_inputs(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    job = load_job(job_id)
    items = iter_input_items(job)
    total = len(items)
    for idx, item in enumerate(items, start=1):
        local_path = ROOT / item["relative_path"]
        update_job(
            job_id,
            progress_message=f"Uploading input {idx}/{total}: {item['stored_name']}",
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
        raise RuntimeError("DUSt3R requires at least two uploaded images.")

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
        f"2>&1 | tee {shlex.quote(log_path)}"
    )

    update_job(
        job_id,
        phase="running_remote_matches",
        progress_message=f"Starting DUSt3R with {n_images} images...",
    )
    _ssh_stream(
        config,
        cmd,
        job_id=job_id,
        phase="running_remote_matches",
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
    if return_code != 0:
        raise RuntimeError(
            f"Remote command failed during {phase}. "
            f"Last message: {last_message or 'No remote log line was captured.'}"
        )


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


def _scp_from_remote(config: ServerConfig, remote_path: str, local_path: Path) -> subprocess.CompletedProcess:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["scp", *SSH_CONNECT_OPTIONS, f"{config.alias}:{remote_path}", str(local_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SCP_TIMEOUT_SECONDS,
    )
