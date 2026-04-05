from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOCAL_JOBS_DIR = ROOT / "local_jobs"


@dataclass
class JobRecord:
    job_id: str
    created_at: str
    model: str
    source_type: str
    notes: str
    status: str = "draft"
    phase: str = "local_prepared"
    input_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    remote_job_dir: str | None = None
    remote_runner: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_local_jobs_dir() -> Path:
    LOCAL_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_JOBS_DIR


def make_job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_job_dir(job_id: str) -> Path:
    return ensure_local_jobs_dir() / job_id


def create_job(model: str, source_type: str, notes: str) -> JobRecord:
    job_id = make_job_id()
    created_at = datetime.now().isoformat(timespec="seconds")
    job = JobRecord(
        job_id=job_id,
        created_at=created_at,
        model=model,
        source_type=source_type,
        notes=notes.strip(),
        remote_runner=_default_runner_for(model),
    )

    job_dir = get_job_dir(job_id)
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_job(job)
    return job


def _default_runner_for(model: str) -> str:
    runners = {
        "dust3r": "dust3r_runner.py",
        "monst3r": "monst3r_runner.py",
    }
    return runners.get(model, "unknown_runner.py")


def save_job(job: JobRecord) -> None:
    job_dir = get_job_dir(job.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(
        json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "status": job.status,
                "phase": job.phase,
                "error_message": job.error_message,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_job(job_id: str) -> JobRecord:
    payload = json.loads((get_job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
    return JobRecord(**payload)


def list_jobs(limit: int = 20) -> list[JobRecord]:
    ensure_local_jobs_dir()
    jobs: list[JobRecord] = []
    for job_json in sorted(LOCAL_JOBS_DIR.glob("*/job.json"), reverse=True):
        payload = json.loads(job_json.read_text(encoding="utf-8"))
        jobs.append(JobRecord(**payload))
        if len(jobs) >= limit:
            break
    return jobs


def save_inputs(job: JobRecord, uploaded_files: Iterable[tuple[str, bytes]]) -> JobRecord:
    job_dir = get_job_dir(job.job_id)
    input_dir = job_dir / "input"
    saved_paths: list[str] = []

    for filename, content in uploaded_files:
        target = input_dir / filename
        target.write_bytes(content)
        saved_paths.append(str(target.relative_to(ROOT)))

    job.input_files = saved_paths
    save_job(job)
    return job


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    phase: str | None = None,
    remote_job_dir: str | None = None,
    output_files: list[str] | None = None,
    error_message: str | None = None,
) -> JobRecord:
    job = load_job(job_id)
    if status is not None:
        job.status = status
    if phase is not None:
        job.phase = phase
    if remote_job_dir is not None:
        job.remote_job_dir = remote_job_dir
    if output_files is not None:
        job.output_files = output_files
    job.error_message = error_message
    save_job(job)
    return job
