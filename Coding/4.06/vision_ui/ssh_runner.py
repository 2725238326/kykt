from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from job_store import ROOT, get_job_dir, load_job, update_job


@dataclass
class ServerConfig:
    alias: str = "KYKT-UI"
    host: str = "172.17.140.97"
    user: str = "kykt26"
    port: int = 22
    remote_root: str = "/hdd3/kykt26"
    remote_jobs_dir: str = "/hdd3/kykt26/jobs"
    remote_dust3r_repo: str = "/hdd3/kykt26/code/dust3r-main"
    remote_dust3r_model: str = "/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    remote_dust3r_env: str = "dust3r"


def describe_next_remote_steps(model: str) -> list[str]:
    if model == "dust3r":
        return [
            "Upload local job input files to /hdd3/kykt26/jobs/<job_id>/input",
            "Upload job.json to the same remote job directory",
            "Run remote dust3r runner through ssh",
            "Download output/result.json, matches image, and point cloud back to local cache",
        ]
    if model == "monst3r":
        return [
            "Upload local frames or video to /hdd3/kykt26/jobs/<job_id>/input",
            "Upload job.json and choose the monst3r runner",
            "Run remote monst3r inference through ssh",
            "Download visualizations, dynamic point cloud, and logs back to local cache",
        ]
    return ["Define a model-specific remote runner before dispatching jobs."]


def run_remote_job(job_id: str) -> None:
    job = load_job(job_id)
    config = ServerConfig()

    try:
        update_job(job_id, status="running", phase="preparing_remote")
        remote_job_dir = f"{config.remote_jobs_dir}/{job.job_id}"
        update_job(job_id, remote_job_dir=remote_job_dir)

        _ssh(
            config,
            (
                f"mkdir -p {shlex.quote(remote_job_dir)}/input "
                f"{shlex.quote(remote_job_dir)}/output "
                f"{shlex.quote(remote_job_dir)}/logs"
            ),
        )

        update_job(job_id, phase="uploading_inputs")
        _upload_inputs(config, job.job_id, remote_job_dir)
        _upload_remote_job_json(config, job.job_id, remote_job_dir)

        if job.model == "dust3r":
            _run_dust3r(config, job.job_id, remote_job_dir)
        else:
            raise RuntimeError(f"Model '{job.model}' is not wired yet.")

        update_job(job_id, phase="downloading_results")
        output_files = _download_results(config, job.job_id, remote_job_dir)
        update_job(job_id, status="finished", phase="finished", output_files=output_files, error_message=None)
    except Exception as exc:
        update_job(job_id, status="failed", phase="failed", error_message=str(exc))


def _upload_inputs(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    job = load_job(job_id)
    for rel_path in job.input_files:
        local_path = ROOT / rel_path
        _scp_to_remote(config, local_path, f"{remote_job_dir}/input/{local_path.name}")


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


def _run_dust3r(config: ServerConfig, job_id: str, remote_job_dir: str) -> None:
    job = load_job(job_id)
    if len(job.input_files) < 2:
        raise RuntimeError("DUSt3R currently requires at least two uploaded images.")

    image_names = [Path(p).name for p in job.input_files[:2]]
    image1 = f"{remote_job_dir}/input/{image_names[0]}"
    image2 = f"{remote_job_dir}/input/{image_names[1]}"
    match_out = f"{remote_job_dir}/output/matches.png"
    ply_out = f"{remote_job_dir}/output/pointcloud.ply"
    matches_log = f"{remote_job_dir}/logs/matches.log"
    pointcloud_log = f"{remote_job_dir}/logs/pointcloud.log"

    usage_cmd = (
        f"cd {shlex.quote(config.remote_dust3r_repo)} && "
        f"conda run -n {shlex.quote(config.remote_dust3r_env)} "
        f"python usage.py "
        f"--model {shlex.quote(config.remote_dust3r_model)} "
        f"--image1 {shlex.quote(image1)} "
        f"--image2 {shlex.quote(image2)} "
        f"--output {shlex.quote(match_out)} "
        f"--n-viz 50 "
        f"> {shlex.quote(matches_log)} 2>&1"
    )

    pointcloud_cmd = (
        f"cd {shlex.quote(config.remote_dust3r_repo)} && "
        f"conda run -n {shlex.quote(config.remote_dust3r_env)} "
        f"python usage_pointcloud.py "
        f"--model {shlex.quote(config.remote_dust3r_model)} "
        f"--image1 {shlex.quote(image1)} "
        f"--image2 {shlex.quote(image2)} "
        f"--output {shlex.quote(ply_out)} "
        f"> {shlex.quote(pointcloud_log)} 2>&1"
    )

    update_job(job_id, phase="running_remote_matches")
    _ssh(config, usage_cmd)
    update_job(job_id, phase="running_remote_pointcloud")
    _ssh(config, pointcloud_cmd)


def _download_results(config: ServerConfig, job_id: str, remote_job_dir: str) -> list[str]:
    job_dir = get_job_dir(job_id)
    local_output_dir = job_dir / "output"
    local_logs_dir = job_dir / "logs"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_logs_dir.mkdir(parents=True, exist_ok=True)

    downloads = [
        ("output/matches.png", local_output_dir / "matches.png"),
        ("output/pointcloud.ply", local_output_dir / "pointcloud.ply"),
        ("logs/matches.log", local_logs_dir / "matches.log"),
        ("logs/pointcloud.log", local_logs_dir / "pointcloud.log"),
    ]

    output_files: list[str] = []
    for remote_suffix, local_path in downloads:
        remote_path = f"{remote_job_dir}/{remote_suffix}"
        _scp_from_remote(config, remote_path, local_path)
        output_files.append(str(local_path.relative_to(ROOT)))
    return output_files


def _ssh(config: ServerConfig, shell_script: str) -> subprocess.CompletedProcess:
    remote_cmd = ["ssh", config.alias, f"bash -lc {shlex.quote(shell_script)}"]
    return subprocess.run(remote_cmd, check=True, capture_output=True, text=True)


def _scp_to_remote(config: ServerConfig, local_path: Path, remote_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["scp", str(local_path), f"{config.alias}:{remote_path}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _scp_from_remote(config: ServerConfig, remote_path: str, local_path: Path) -> subprocess.CompletedProcess:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["scp", f"{config.alias}:{remote_path}", str(local_path)],
        check=True,
        capture_output=True,
        text=True,
    )
