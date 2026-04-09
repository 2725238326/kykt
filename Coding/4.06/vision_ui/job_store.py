from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOCAL_JOBS_DIR = ROOT / "local_jobs"
_JOB_STORE_LOCK = threading.RLock()


@dataclass
class JobRecord:
    job_id: str
    created_at: str
    model: str
    source_type: str
    notes: str
    params: dict = field(default_factory=dict)
    status: str = "draft"
    phase: str = "local_prepared"
    input_files: list[str] = field(default_factory=list)
    input_items: list[dict] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    remote_job_dir: str | None = None
    remote_runner: str | None = None
    error_message: str | None = None
    progress_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_local_jobs_dir() -> Path:
    LOCAL_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_JOBS_DIR


def make_job_id() -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base
    suffix = 1
    while get_job_dir(candidate).exists():
        candidate = f"{base}-{suffix:02d}"
        suffix += 1
    return candidate


def get_job_dir(job_id: str) -> Path:
    return ensure_local_jobs_dir() / job_id


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def create_job(model: str, source_type: str, notes: str, params: dict | None = None) -> JobRecord:
    job_id = make_job_id()
    created_at = datetime.now().isoformat(timespec="seconds")
    job = JobRecord(
        job_id=job_id,
        created_at=created_at,
        model=model,
        source_type=source_type,
        notes=notes.strip(),
        params=params or {},
        remote_runner=_default_runner_for(model),
        progress_message="本地任务已就绪。文件名没有要求，系统会自动规范化内部命名。",
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
    with _JOB_STORE_LOCK:
        job_dir = get_job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_json(job_dir / "job.json", job.to_dict())
        _write_json(
            job_dir / "status.json",
            {
                "status": job.status,
                "phase": job.phase,
                "error_message": job.error_message,
                "progress_message": job.progress_message,
            },
        )


def load_job(job_id: str) -> JobRecord:
    with _JOB_STORE_LOCK:
        payload = _read_json(get_job_dir(job_id) / "job.json")
        payload.setdefault("params", {})
        payload.setdefault("input_items", [])
        return JobRecord(**payload)


def list_jobs(limit: int = 20) -> list[JobRecord]:
    with _JOB_STORE_LOCK:
        ensure_local_jobs_dir()
        jobs: list[JobRecord] = []
        for job_json in sorted(LOCAL_JOBS_DIR.glob("*/job.json"), reverse=True):
            payload = _read_json(job_json)
            payload.setdefault("params", {})
            payload.setdefault("input_items", [])
            jobs.append(JobRecord(**payload))
            if len(jobs) >= limit:
                break
        return jobs


def _normalized_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix else ".bin"


def iter_input_items(job: JobRecord) -> list[dict]:
    if job.input_items:
        return list(job.input_items)

    items = []
    for rel_path in job.input_files:
        path = Path(rel_path)
        items.append(
            {
                "original_name": path.name,
                "stored_name": path.name,
                "relative_path": rel_path,
                "size_bytes": None,
            }
        )
    return items


def save_inputs(job: JobRecord, uploaded_files: Iterable[tuple[str, bytes]]) -> JobRecord:
    uploads = list(uploaded_files)
    job_dir = get_job_dir(job.job_id)
    input_dir = job_dir / "input"
    saved_paths: list[str] = []
    saved_items: list[dict] = []

    width = max(2, len(str(max(len(uploads), 1))))

    for index, (filename, content) in enumerate(uploads, start=1):
        suffix = _normalized_suffix(filename)
        stored_name = f"input_{index:0{width}d}{suffix}"
        target = input_dir / stored_name
        target.write_bytes(content)
        relative_path = str(target.relative_to(ROOT))
        saved_paths.append(relative_path)
        saved_items.append(
            {
                "original_name": filename,
                "stored_name": stored_name,
                "relative_path": relative_path,
                "size_bytes": len(content),
            }
        )

    job.input_files = saved_paths
    job.input_items = saved_items
    save_job(job)
    return job


def duplicate_job(job_id: str) -> JobRecord:
    source = load_job(job_id)
    new_job = create_job(
        model=source.model,
        source_type=source.source_type,
        notes=source.notes,
        params=dict(source.params),
    )

    uploads = []
    for item in iter_input_items(source):
        local_path = ROOT / item["relative_path"]
        uploads.append((item["original_name"], local_path.read_bytes()))

    save_inputs(new_job, uploads)
    update_job(
        new_job.job_id,
        progress_message=f"已从 {job_id} 复制任务，输入文件保持一致，可以直接运行。",
    )
    return load_job(new_job.job_id)


def clear_job_runtime(job_id: str) -> JobRecord:
    job = load_job(job_id)
    job_dir = get_job_dir(job_id)

    for folder_name in ("output", "logs"):
        folder = job_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for child in folder.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    remote_payload = job_dir / "remote_job.json"
    if remote_payload.exists():
        remote_payload.unlink()

    job.status = "draft"
    job.phase = "local_prepared"
    job.output_files = []
    job.error_message = None
    job.progress_message = "任务已在本地重置，可以重新调度。"
    save_job(job)
    return job


def get_log_snippets(job_id: str, limit: int = 60) -> list[dict]:
    snippets: list[dict] = []
    logs_dir = get_job_dir(job_id) / "logs"
    if not logs_dir.exists():
        return snippets

    for log_path in sorted(logs_dir.glob("*.log")):
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        except OSError:
            tail = []
        snippets.append(
            {
                "name": log_path.name,
                "relative_path": str(log_path.relative_to(ROOT)),
                "tail": "\n".join(tail),
            }
        )
    return snippets


def _render_summary_markdown(payload: dict) -> str:
    lines = [
        f"# 任务摘要：{payload.get('job_id', 'unknown')}",
        "",
        f"- 模型：{payload.get('model', 'unknown')}",
        f"- 状态：{payload.get('status_label', payload.get('status', 'unknown'))}",
        f"- 输入类型：{payload.get('source_type', 'unknown')}",
        f"- 生成时间：{payload.get('generated_at', '-')}",
        "",
    ]

    highlights = payload.get("highlights") or []
    if highlights:
        lines.extend(["## 关键结果", ""])
        lines.extend([f"- {item}" for item in highlights])
        lines.append("")

    params = payload.get("params") or {}
    if params:
        lines.extend(["## 运行参数", ""])
        lines.extend([f"- {key}: {value}" for key, value in params.items()])
        lines.append("")

    artifacts = payload.get("artifacts") or []
    if artifacts:
        lines.extend(["## 产物列表", ""])
        for item in artifacts:
            lines.append(f"- {item.get('name', 'unknown')}: {item.get('relative_path', '-')}")
        lines.append("")

    next_actions = payload.get("next_actions") or []
    if next_actions:
        lines.extend(["## 建议后续动作", ""])
        lines.extend([f"- {item}" for item in next_actions])
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_result_summary(job_id: str, payload: dict) -> None:
    job_dir = get_job_dir(job_id)
    _write_json(job_dir / "result_summary.json", payload)
    (job_dir / "result_summary.md").write_text(_render_summary_markdown(payload), encoding="utf-8")


def load_result_summary(job_id: str) -> dict | None:
    path = get_job_dir(job_id) / "result_summary.json"
    if not path.exists():
        return None
    return _read_json(path)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    phase: str | None = None,
    remote_job_dir: str | None = None,
    output_files: list[str] | None = None,
    error_message: str | None = None,
    progress_message: str | None = None,
) -> JobRecord:
    with _JOB_STORE_LOCK:
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
        if progress_message is not None:
            job.progress_message = progress_message
        save_job(job)
        return job
