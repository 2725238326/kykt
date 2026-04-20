from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from advisor import (
    advisor_config_public,
    advisor_status,
    evaluate_job_with_advisor,
    load_advisor_report,
    save_advisor_config,
)
from job_store import (
    EVALUATION_SCORE_FIELDS,
    EVALUATION_SCORE_MAX,
    EVALUATION_SCORE_MIN,
    ROOT,
    clear_job_runtime,
    create_job,
    duplicate_job,
    get_log_snippets,
    iter_input_items,
    list_jobs,
    load_evaluation,
    load_result_summary,
    load_job,
    save_inputs,
    save_evaluation,
    update_job,
)
from model_registry import MODEL_CATALOG_OPTIONS, MODEL_OPTIONS, SOURCE_TYPE_OPTIONS, allowed_source_types, get_model_spec
from ssh_runner import ServerConfig, cancel_remote_job, run_remote_job

_RUNNER_THREADS: dict[str, threading.Thread] = {}
_RUNNER_THREADS_LOCK = threading.Lock()
_SAMPLES_CACHE_LOCK = threading.RLock()
_SAMPLES_CACHE: tuple[int | None, dict] | None = None
_DEPLOYMENT_STATUS_CACHE_LOCK = threading.Condition(threading.RLock())
_DEPLOYMENT_STATUS_CACHE: dict | None = None
_DEPLOYMENT_STATUS_REFRESHING = False
DEPLOYMENT_STATUS_TTL_SECONDS = 20.0
DEPLOYMENT_STATUS_STALE_SECONDS = 300.0
DEPLOYMENT_STATUS_TIMEOUT_SECONDS = 15.0


app = FastAPI(title="KYKT Vision UI", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:1420",
        "http://localhost:1420",
        "http://tauri.localhost",
        "tauri://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.globals["asset_version"] = "20260410-2130"
SAMPLES_MANIFEST_PATH = ROOT / "samples_manifest.json"
DEPLOYMENT_SCRIPT_PATH = ROOT.parents[2] / "tools" / "check_3r_remote.ps1"

(ROOT / "static").mkdir(parents=True, exist_ok=True)
(ROOT / "local_jobs").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/local_jobs", StaticFiles(directory=str(ROOT / "local_jobs")), name="local_jobs")


PHASE_FLOW = [
    ("local_prepared", "本地任务已就绪", "本地任务记录和输入缓存已经准备好。", 4, 8),
    ("preparing_remote", "准备服务器目录", "正在创建远端任务目录和任务文件。", 8, 15),
    ("uploading_inputs", "上传输入文件", "正在把输入文件和任务清单发送到服务器。", 15, 25),
    ("running_remote_matches", "运行模型推理与重建", "正在执行远端模型推理、匹配或序列重建流程。", 25, 70),
    ("running_remote_pointcloud", "整理三维产物", "正在导出点云、三维场景或其他远端输出文件。", 70, 90),
    ("downloading_results", "下载结果", "正在把输出文件和日志拉回本地缓存。", 90, 98),
    ("finished", "已完成", "任务已成功完成。", 100, 100),
    ("failed", "失败", "任务因错误停止，请查看日志后重试。", 0, 0),
    ("cancelled", "已取消", "任务已在本地取消，必要时请检查服务器端是否还有残留进程。", 0, 0),
]

STATUS_LABELS = {
    "created": "已创建",
    "ready": "已就绪",
    "running": "运行中",
    "finished": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}

DELIVERY_GAPS = [
    {
        "title": "主动新模型还缺环境和官方 smoke run",
        "detail": "Spann3R、Align3R、Fast3R、CUT3R 目录已就绪，但还需要独立 conda env、官方 repo、权重和 first smoke run。",
    },
    {
        "title": "远端取消与清理仍然不够硬",
        "detail": "现在可以本地标记取消并尝试 pkill，但还缺更可靠的远端进程确认和残留目录清理。",
    },
    {
        "title": "模型间对比还缺评分闭环",
        "detail": "样例库和测评矩阵已经有雏形，但还缺每个任务的人工评分、同样例结果对比和最终报告导出。",
    },
    {
        "title": "结果归档仍然不够完整",
        "detail": "现在已经会自动生成任务摘要，但还缺更正式的交付打包、汇总报告和归档策略。",
    },
    {
        "title": "交互恢复能力还需要加强",
        "detail": "Windows 侧旧 uvicorn/ssh 进程卡住时，仍然需要更明确的检测、提示和一键恢复动作。",
    },
]

ACTIVE_PHASE_CODES = [code for code, *_ in PHASE_FLOW[:6]]
PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")
EVALUATION_FIELD_LABELS = {
    "structure_completeness": "结构完整性",
    "trajectory_stability": "轨迹稳定性",
    "noise": "噪声",
    "dynamic_handling": "动态处理",
    "depth_continuity": "深度连续性",
    "presentation_usability": "展示可用性",
}
EVALUATION_FIELD_ALIASES = {
    "noise_control": "noise",
    "depth_consistency": "depth_continuity",
}


def status_label(status: str | None) -> str:
    if not status:
        return "未知"
    return STATUS_LABELS.get(status, status)


templates.env.globals["status_label"] = status_label


def build_dashboard_stats(jobs) -> dict:
    summary = {
        "total": len(jobs),
        "running": 0,
        "finished": 0,
        "failed": 0,
        "cancelled": 0,
    }
    for job in jobs:
        key = job.status if job.status in summary else None
        if key:
            summary[key] += 1
    return summary


def load_samples_manifest() -> dict:
    global _SAMPLES_CACHE

    if not SAMPLES_MANIFEST_PATH.exists():
        return {
            "last_updated": None,
            "purpose": "Shared sample plan has not been created yet.",
            "active_models": [],
            "deferred_models": [],
            "samples": [],
            "scoring": {},
        }

    try:
        mtime_ns = SAMPLES_MANIFEST_PATH.stat().st_mtime_ns
        with _SAMPLES_CACHE_LOCK:
            if _SAMPLES_CACHE and _SAMPLES_CACHE[0] == mtime_ns:
                return _SAMPLES_CACHE[1]
            payload = json.loads(SAMPLES_MANIFEST_PATH.read_text(encoding="utf-8-sig"))
            _SAMPLES_CACHE = (mtime_ns, payload)
            return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"样例清单读取失败：{exc}") from exc


def build_sample_status_summary(manifest: dict) -> dict:
    samples = manifest.get("samples") or []
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    required_model_counts: dict[str, int] = {}

    for sample in samples:
        status = str(sample.get("status") or "unknown")
        source_type = str(sample.get("source_type") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        for model in sample.get("required_models") or []:
            model_key = str(model)
            required_model_counts[model_key] = required_model_counts.get(model_key, 0) + 1

    return {
        "sample_count": len(samples),
        "status_counts": status_counts,
        "source_counts": source_counts,
        "required_model_counts": required_model_counts,
    }


def _extract_progress_ratio(progress_message: str | None) -> float | None:
    if not progress_message:
        return None

    matches = PROGRESS_PATTERN.findall(progress_message)
    for done_str, total_str in reversed(matches):
        done = int(done_str)
        total = int(total_str)
        if total > 0 and 0 <= done <= total:
            return done / total
    return None


def build_phase_display(phase: str, status: str, progress_message: str | None = None) -> dict:
    known_phases = {code: (label, hint, start, end) for code, label, hint, start, end in PHASE_FLOW}
    if phase not in known_phases:
        phase = "local_prepared"

    label, description, start, end = known_phases[phase]
    ratio = _extract_progress_ratio(progress_message)

    if status == "finished":
        percent = 100
    elif phase == "failed":
        percent = 100 if status == "finished" else 0
    elif ratio is not None and end > start:
        percent = min(end, max(start, int(start + (end - start) * ratio)))
    elif phase == "running_remote_matches":
        percent = 40
    elif phase == "running_remote_pointcloud":
        percent = 80
    else:
        percent = end

    steps = []
    if phase in ACTIVE_PHASE_CODES:
        current_index = ACTIVE_PHASE_CODES.index(phase)
    elif status == "cancelled":
        current_index = 0
    else:
        current_index = len(ACTIVE_PHASE_CODES)
    for idx, code in enumerate(ACTIVE_PHASE_CODES):
        item_label, item_hint, *_ = known_phases[code]
        state = "todo"
        if status == "finished":
            state = "done"
        elif status == "failed":
            if idx < current_index:
                state = "done"
            elif idx == current_index:
                state = "current"
        elif idx < current_index:
            state = "done"
        elif idx == current_index:
            state = "current"
        steps.append({"code": code, "label": item_label, "hint": item_hint, "state": state})

    return {
        "label": label,
        "description": description,
        "percent": percent,
        "steps": steps,
    }


def serialize_outputs(job) -> list[dict]:
    outputs = []
    for rel_path in job.output_files:
        suffix = Path(rel_path).suffix.lower()
        if suffix in {".json", ".log"}:
            continue
        outputs.append(
            {
                "relative_path": rel_path,
                "display_name": Path(rel_path).name,
                "url": "/" + rel_path.replace("\\", "/"),
                "is_image": suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
                "is_pointcloud": suffix == ".ply",
                "is_model3d": suffix in {".glb", ".gltf"},
                "is_video": suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"},
                "is_log": suffix == ".log",
            }
        )
    return outputs


def resolve_local_output(job, relative_path: str) -> Path:
    if relative_path not in job.output_files:
        raise HTTPException(status_code=404, detail="任务中没有这个输出文件。")

    root = ROOT.resolve()
    target = (ROOT / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="输出文件路径不合法。") from exc

    if not target.exists():
        raise HTTPException(status_code=404, detail="本地输出文件不存在。")
    return target


def serialize_previews(job) -> list[dict]:
    previews = []
    for item in iter_input_items(job):
        rel_path = item["relative_path"]
        suffix = Path(rel_path).suffix.lower()
        previews.append(
            {
                "relative_path": rel_path,
                "display_name": item["original_name"],
                "stored_name": item["stored_name"],
                "url": "/" + rel_path.replace("\\", "/"),
                "is_image": suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
            }
        )
    return previews


def _job_payload(job) -> dict:
    return {
        "job": job.to_dict(),
        "phase_display": build_phase_display(job.phase, job.status, job.progress_message),
        "outputs": serialize_outputs(job),
        "previews": serialize_previews(job),
        "logs": get_log_snippets(job.job_id),
        "result_summary": load_result_summary(job.job_id),
        "evaluation": load_evaluation(job.job_id),
        "advisor_report": load_advisor_report(job.job_id),
    }


def _parse_evaluation_score(field_name: str, raw_value) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
    if isinstance(raw_value, bool):
        raise HTTPException(status_code=400, detail=f"{EVALUATION_FIELD_LABELS[field_name]} 必须是整数分数。")
    if isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise HTTPException(status_code=400, detail=f"{EVALUATION_FIELD_LABELS[field_name]} 必须是整数分数。")
        raw_value = int(raw_value)
    elif isinstance(raw_value, str):
        try:
            raw_value = int(raw_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{EVALUATION_FIELD_LABELS[field_name]} 必须是整数分数。") from exc
    elif not isinstance(raw_value, int):
        raise HTTPException(status_code=400, detail=f"{EVALUATION_FIELD_LABELS[field_name]} 必须是整数分数。")

    if raw_value < EVALUATION_SCORE_MIN or raw_value > EVALUATION_SCORE_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"{EVALUATION_FIELD_LABELS[field_name]} 必须在 {EVALUATION_SCORE_MIN} 到 {EVALUATION_SCORE_MAX} 分之间。",
        )
    return raw_value


def _normalize_evaluation_payload(job_id: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="评分请求体必须是 JSON 对象。")

    normalized = load_evaluation(job_id)
    score_source = payload.get("scores")
    if score_source is not None and not isinstance(score_source, dict):
        raise HTTPException(status_code=400, detail="scores 字段必须是对象。")
    score_source = score_source if isinstance(score_source, dict) else {}

    for field_name in EVALUATION_SCORE_FIELDS:
        alias_name = next((alias for alias, canonical in EVALUATION_FIELD_ALIASES.items() if canonical == field_name), None)
        if field_name in payload:
            raw_value = payload[field_name]
        elif alias_name and alias_name in payload:
            raw_value = payload[alias_name]
        elif field_name in score_source:
            raw_value = score_source[field_name]
        elif alias_name and alias_name in score_source:
            raw_value = score_source[alias_name]
        else:
            continue
        normalized[field_name] = _parse_evaluation_score(field_name, raw_value)

    if "notes" in payload:
        notes = payload["notes"]
        if notes is None:
            normalized["notes"] = ""
        elif isinstance(notes, str):
            normalized["notes"] = notes.strip()
        else:
            normalized["notes"] = str(notes).strip()

    return normalized


def _dust3r_params(
    image_size: int,
    scene_graph: str,
    niter: int,
    lr: float,
    batch_size: int,
    max_points: int,
    match_viz_count: int,
) -> dict:
    return {
        "image_size": min(max(int(image_size), 224), 1024),
        "scene_graph": scene_graph.strip() or "complete",
        "niter": min(max(int(niter), 0), 1000),
        "lr": max(float(lr), 0.0),
        "batch_size": min(max(int(batch_size), 1), 8),
        "max_points": min(max(int(max_points), 1000), 2_000_000),
        "match_viz_count": min(max(int(match_viz_count), 0), 500),
    }


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _monst3r_params(
    image_size: int,
    batch_size: int,
    fps: int,
    num_frames: int,
    not_batchify: str | bool,
    real_time: str | bool,
    window_wise: str | bool,
    window_size: int,
    window_overlap_ratio: float,
) -> dict:
    normalized_image_size = int(image_size)
    if normalized_image_size not in {224, 512}:
        normalized_image_size = 512
    return {
        "image_size": normalized_image_size,
        "batch_size": min(max(int(batch_size), 1), 16),
        "fps": min(max(int(fps), 0), 120),
        "num_frames": min(max(int(num_frames), 1), 2000),
        "not_batchify": _parse_bool(not_batchify, True),
        "real_time": _parse_bool(real_time, False),
        "window_wise": _parse_bool(window_wise, False),
        "window_size": min(max(int(window_size), 2), 500),
        "window_overlap_ratio": min(max(float(window_overlap_ratio), 0.0), 0.95),
    }


def _validate_new_job(model: str, source_type: str, files: list[UploadFile]) -> None:
    model_values = {item["value"] for item in MODEL_OPTIONS}
    source_values = {item["value"] for item in SOURCE_TYPE_OPTIONS}
    if model not in model_values:
        raise HTTPException(status_code=400, detail=f"不支持的模型：{model}")
    if source_type not in source_values:
        raise HTTPException(status_code=400, detail=f"不支持的输入类型：{source_type}")
    if source_type not in set(allowed_source_types(model)):
        allowed = " / ".join(allowed_source_types(model))
        raise HTTPException(status_code=400, detail=f"{get_model_spec(model).label} 仅支持这些输入类型：{allowed}")
    if not files:
        raise HTTPException(status_code=400, detail="没有上传输入文件。")
    if model in {"dust3r", "mast3r"} and len(files) < 2:
        raise HTTPException(status_code=400, detail=f"{get_model_spec(model).label} 至少需要两张输入图片。")
    if model == "monst3r" and source_type == "video" and len(files) != 1:
        raise HTTPException(status_code=400, detail="MonST3R 视频模式请上传 1 个视频文件；多张图片请改选“帧序列”。")


def _validate_dispatchable(job) -> None:
    if job.model in {"dust3r", "mast3r"} and len(job.input_files) < 2:
        raise HTTPException(status_code=400, detail=f"{get_model_spec(job.model).label} 至少需要两张输入图片。")
    if job.model == "monst3r" and len(job.input_files) < 1:
        raise HTTPException(status_code=400, detail="MonST3R 至少需要 1 个视频或一组帧序列。")


def _runner_thread_target(job_id: str) -> None:
    try:
        run_remote_job(job_id)
    finally:
        with _RUNNER_THREADS_LOCK:
            existing = _RUNNER_THREADS.get(job_id)
            if existing is threading.current_thread():
                _RUNNER_THREADS.pop(job_id, None)


def _launch_remote_job(job_id: str) -> None:
    with _RUNNER_THREADS_LOCK:
        existing = _RUNNER_THREADS.get(job_id)
        if existing and existing.is_alive():
            raise HTTPException(status_code=409, detail=f"任务 {job_id} 已经在后台运行。")
        thread = threading.Thread(
            target=_runner_thread_target,
            args=(job_id,),
            daemon=True,
            name=f"vision-remote-job-{job_id}",
        )
        _RUNNER_THREADS[job_id] = thread
        thread.start()


@app.get("/")
async def index(request: Request):
    jobs = list_jobs(limit=50)
    summary = build_dashboard_stats(jobs)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": jobs,
            "summary": summary,
            "delivery_gaps": DELIVERY_GAPS,
            "server": ServerConfig(),
            "models": [(item["value"], f"{item['label']}（{item['description']}）") for item in MODEL_OPTIONS],
            "phase_builder": build_phase_display,
        },
    )


@app.post("/jobs")
async def create_job_view(
    model: str = Form(...),
    source_type: str = Form(...),
    notes: str = Form(""),
    image_size: int = Form(512),
    scene_graph: str = Form("complete"),
    niter: int = Form(300),
    lr: float = Form(0.01),
    batch_size: int = Form(1),
    max_points: int = Form(250000),
    match_viz_count: int = Form(50),
    fps: int = Form(0),
    num_frames: int = Form(24),
    not_batchify: str = Form("true"),
    real_time: str = Form("false"),
    window_wise: str = Form("false"),
    window_size: int = Form(100),
    window_overlap_ratio: float = Form(0.5),
    files: list[UploadFile] = File(...),
):
    _validate_new_job(model, source_type, files)

    params = {}
    if model in {"dust3r", "mast3r"}:
        params = _dust3r_params(image_size, scene_graph, niter, lr, batch_size, max_points, match_viz_count)
    elif model == "monst3r":
        params = _monst3r_params(
            image_size,
            batch_size,
            fps,
            num_frames,
            not_batchify,
            real_time,
            window_wise,
            window_size,
            window_overlap_ratio,
        )

    job = create_job(model=model, source_type=source_type, notes=notes, params=params)
    uploaded = []
    for upload in files:
        uploaded.append((upload.filename or "unnamed.bin", await upload.read()))
    save_inputs(job, uploaded)

    return RedirectResponse(url=f"/jobs/{job.job_id}", status_code=303)


@app.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    payload = _job_payload(job)
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            **payload,
            "job": job,
        },
    )


@app.post("/jobs/{job_id}/dispatch")
async def dispatch_job(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    _validate_dispatchable(job)

    clear_job_runtime(job_id)
    update_job(
        job_id,
        status="running",
        phase="preparing_remote",
        error_message=None,
        progress_message="正在启动远端调度线程...",
    )
    _launch_remote_job(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    _validate_dispatchable(job)

    clear_job_runtime(job_id)
    update_job(
        job_id,
        status="running",
        phase="preparing_remote",
        error_message=None,
        progress_message="正在重新启动远端调度线程...",
    )
    _launch_remote_job(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/duplicate")
async def duplicate_job_view(job_id: str):
    try:
        new_job = duplicate_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    return RedirectResponse(url=f"/jobs/{new_job.job_id}", status_code=303)


@app.post("/jobs/{job_id}/mark-failed")
async def mark_job_failed(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    update_job(
        job_id,
        status="failed",
        phase="failed",
        error_message="用户已在本地将任务标记为失败。未尝试清理远端进程。",
        progress_message="已在本地标记为失败。可以点击重试重新调度。",
    )
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job_view(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    cancel_remote_job(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/open-output")
async def open_output_file(job_id: str, relative_path: str = Form(...)):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    target = resolve_local_output(job, relative_path)
    try:
        os.startfile(str(target))  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise HTTPException(status_code=400, detail="当前系统不支持用默认程序打开本地文件。") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"打开本地文件失败：{exc}") from exc

    return JSONResponse({"ok": True, "path": str(target)})


@app.post("/api/jobs/{job_id}/open-output")
async def open_output_file_api(job_id: str, relative_path: str = Form(...)):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    target = resolve_local_output(job, relative_path)
    try:
        os.startfile(str(target))  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise HTTPException(status_code=400, detail="当前系统不支持用默认程序打开本地文件。") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"打开本地文件失败：{exc}") from exc

    return JSONResponse({"ok": True, "path": str(target)})


@app.get("/api/jobs")
async def jobs_api():
    jobs = list_jobs(limit=50)
    payload = []
    for job in jobs:
        payload.append(
            {
                "job": job.to_dict(),
                "phase_display": build_phase_display(job.phase, job.status, job.progress_message),
            }
        )
    return JSONResponse({"jobs": payload, "summary": build_dashboard_stats(jobs)})


@app.get("/api/jobs/{job_id}")
async def job_detail_api(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    return JSONResponse(_job_payload(job))


@app.get("/api/health")
async def health_api():
    return JSONResponse({"ok": True, "service": "kykt-vision-ui", "version": app.version})


def load_deployment_status(*, force_refresh: bool = False) -> dict:
    global _DEPLOYMENT_STATUS_CACHE, _DEPLOYMENT_STATUS_REFRESHING

    def _utc_iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    def _cache_age_seconds(entry: dict, now_mono: float) -> float:
        return max(0.0, now_mono - float(entry["fetched_monotonic"]))

    def _build_response(entry: dict, *, state: str, error: str | None = None) -> dict:
        age_seconds = _cache_age_seconds(entry, time.monotonic())
        payload = copy.deepcopy(entry["payload"])
        payload["ok"] = bool((payload.get("summary") or {}).get("ok"))
        payload["source"] = state
        payload["stale"] = state.startswith("stale")
        payload["fetched_at"] = entry["fetched_at"]
        payload["cache"] = {
            "state": state,
            "hit": state != "live",
            "age_seconds": round(age_seconds, 3),
            "ttl_seconds": DEPLOYMENT_STATUS_TTL_SECONDS,
            "stale_ttl_seconds": DEPLOYMENT_STATUS_STALE_SECONDS,
            "timeout_seconds": DEPLOYMENT_STATUS_TIMEOUT_SECONDS,
            "expires_at": _utc_iso(entry["fetched_wall_time"] + DEPLOYMENT_STATUS_TTL_SECONDS),
            "script_path": str(DEPLOYMENT_SCRIPT_PATH),
            "ssh_alias": ServerConfig.alias,
        }
        if error:
            payload["cache"]["last_error"] = error
        return payload

    def _parse_deployment_payload(stdout: str) -> dict:
        stripped = stdout.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(stripped[start : end + 1])

    def _run_status_command() -> dict:
        if not DEPLOYMENT_SCRIPT_PATH.exists():
            raise HTTPException(status_code=500, detail=f"远端部署检查脚本不存在：{DEPLOYMENT_SCRIPT_PATH}")

        powershell_executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        command = [
            powershell_executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEPLOYMENT_SCRIPT_PATH),
            "-SshAlias",
            ServerConfig.alias,
            "-Json",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=DEPLOYMENT_STATUS_TIMEOUT_SECONDS,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="远端部署状态检查超时。") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or "远端部署状态检查失败。"
            raise HTTPException(status_code=502, detail=detail) from exc

        try:
            return _parse_deployment_payload(completed.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="远端部署状态返回了无法解析的 JSON。") from exc

    now_mono = time.monotonic()
    with _DEPLOYMENT_STATUS_CACHE_LOCK:
        while _DEPLOYMENT_STATUS_REFRESHING:
            entry = _DEPLOYMENT_STATUS_CACHE
            if entry and _cache_age_seconds(entry, now_mono) < DEPLOYMENT_STATUS_STALE_SECONDS:
                return _build_response(entry, state="stale-refreshing")
            _DEPLOYMENT_STATUS_CACHE_LOCK.wait(timeout=0.25)
            now_mono = time.monotonic()

        entry = _DEPLOYMENT_STATUS_CACHE
        if entry and not force_refresh and _cache_age_seconds(entry, now_mono) < DEPLOYMENT_STATUS_TTL_SECONDS:
            return _build_response(entry, state="cache")

        _DEPLOYMENT_STATUS_REFRESHING = True

    try:
        payload = _run_status_command()
    except HTTPException as exc:
        with _DEPLOYMENT_STATUS_CACHE_LOCK:
            _DEPLOYMENT_STATUS_REFRESHING = False
            _DEPLOYMENT_STATUS_CACHE_LOCK.notify_all()
            entry = _DEPLOYMENT_STATUS_CACHE
            if entry and _cache_age_seconds(entry, time.monotonic()) < DEPLOYMENT_STATUS_STALE_SECONDS:
                return _build_response(entry, state="stale-error", error=str(exc.detail))
        raise

    fetched_wall_time = time.time()
    entry = {
        "payload": payload,
        "fetched_at": _utc_iso(fetched_wall_time),
        "fetched_wall_time": fetched_wall_time,
        "fetched_monotonic": time.monotonic(),
    }
    with _DEPLOYMENT_STATUS_CACHE_LOCK:
        _DEPLOYMENT_STATUS_CACHE = entry
        _DEPLOYMENT_STATUS_REFRESHING = False
        _DEPLOYMENT_STATUS_CACHE_LOCK.notify_all()
    return _build_response(entry, state="live")


@app.get("/api/bootstrap")
async def bootstrap_api():
    jobs = list_jobs(limit=50)
    return JSONResponse(
        {
            "summary": build_dashboard_stats(jobs),
            "delivery_gaps": DELIVERY_GAPS,
            "server": {
                "alias": ServerConfig.alias,
                "host": ServerConfig.host,
                "user": ServerConfig.user,
                "port": ServerConfig.port,
                "remote_root": ServerConfig.remote_root,
            },
            "models": MODEL_OPTIONS,
            "model_catalog": MODEL_CATALOG_OPTIONS,
            "source_types": SOURCE_TYPE_OPTIONS,
            "advisor": advisor_status(),
        }
    )


@app.get("/api/deployment/status")
async def deployment_status_api(refresh: bool = False):
    return JSONResponse(await asyncio.to_thread(load_deployment_status, force_refresh=refresh))


@app.get("/api/samples")
async def samples_api():
    manifest = load_samples_manifest()
    return JSONResponse(
        {
            "manifest": manifest,
            "summary": build_sample_status_summary(manifest),
            "model_catalog": MODEL_CATALOG_OPTIONS,
        }
        )


@app.get("/api/jobs/{job_id}/evaluation")
async def job_evaluation_api(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc
    return JSONResponse(load_evaluation(job_id))


@app.post("/api/jobs/{job_id}/evaluation")
async def save_job_evaluation_api(job_id: str, request: Request):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"评分 JSON 解析失败：{exc.msg}") from exc

    saved = save_evaluation(job_id, _normalize_evaluation_payload(job_id, payload))
    return JSONResponse({"ok": True, "evaluation": saved, **_job_payload(load_job(job_id))})


@app.post("/api/jobs")
async def create_job_api(
    model: str = Form(...),
    source_type: str = Form(...),
    notes: str = Form(""),
    image_size: int = Form(512),
    scene_graph: str = Form("complete"),
    niter: int = Form(300),
    lr: float = Form(0.01),
    batch_size: int = Form(1),
    max_points: int = Form(250000),
    match_viz_count: int = Form(50),
    fps: int = Form(0),
    num_frames: int = Form(24),
    not_batchify: str = Form("true"),
    real_time: str = Form("false"),
    window_wise: str = Form("false"),
    window_size: int = Form(100),
    window_overlap_ratio: float = Form(0.5),
    files: list[UploadFile] = File(...),
):
    _validate_new_job(model, source_type, files)

    params = {}
    if model in {"dust3r", "mast3r"}:
        params = _dust3r_params(image_size, scene_graph, niter, lr, batch_size, max_points, match_viz_count)
    elif model == "monst3r":
        params = _monst3r_params(
            image_size,
            batch_size,
            fps,
            num_frames,
            not_batchify,
            real_time,
            window_wise,
            window_size,
            window_overlap_ratio,
        )

    job = create_job(model=model, source_type=source_type, notes=notes, params=params)
    uploaded = []
    for upload in files:
        uploaded.append((upload.filename or "unnamed.bin", await upload.read()))
    save_inputs(job, uploaded)
    return JSONResponse(_job_payload(load_job(job.job_id)))


@app.get("/api/advisor/status")
async def advisor_status_api():
    return JSONResponse(advisor_status())


@app.get("/api/advisor/config")
async def advisor_config_api():
    return JSONResponse(advisor_config_public())


@app.post("/api/advisor/config")
async def advisor_config_save_api(request: Request):
    payload = await request.json()
    try:
        config = save_advisor_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"AI 配置保存失败：{exc}") from exc
    return JSONResponse(config)


@app.post("/api/jobs/{job_id}/dispatch")
async def dispatch_job_api(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    _validate_dispatchable(job)

    clear_job_runtime(job_id)
    update_job(
        job_id,
        status="running",
        phase="preparing_remote",
        error_message=None,
        progress_message="正在启动远端调度线程...",
    )
    _launch_remote_job(job_id)
    return JSONResponse({"ok": True, **_job_payload(load_job(job_id))})


@app.post("/api/jobs/{job_id}/retry")
async def retry_job_api(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    _validate_dispatchable(job)

    clear_job_runtime(job_id)
    update_job(
        job_id,
        status="running",
        phase="preparing_remote",
        error_message=None,
        progress_message="正在重新启动远端调度线程...",
    )
    _launch_remote_job(job_id)
    return JSONResponse({"ok": True, **_job_payload(load_job(job_id))})


@app.post("/api/jobs/{job_id}/duplicate")
async def duplicate_job_api(job_id: str):
    try:
        new_job = duplicate_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    return JSONResponse({"ok": True, **_job_payload(load_job(new_job.job_id))})


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_api(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    cancel_remote_job(job_id)
    return JSONResponse({"ok": True, **_job_payload(load_job(job_id))})


@app.post("/api/jobs/{job_id}/advisor/evaluate")
async def advisor_evaluate_api(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    try:
        evaluate_job_with_advisor(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"ok": True, **_job_payload(load_job(job_id))})
