from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from advisor import advisor_status, evaluate_job_with_advisor, load_advisor_report
from job_store import (
    ROOT,
    clear_job_runtime,
    create_job,
    duplicate_job,
    get_log_snippets,
    iter_input_items,
    list_jobs,
    load_result_summary,
    load_job,
    save_inputs,
    update_job,
)
from model_registry import MODEL_OPTIONS, SOURCE_TYPE_OPTIONS, allowed_source_types, get_model_spec
from ssh_runner import ServerConfig, cancel_remote_job, run_remote_job

_RUNNER_THREADS: dict[str, threading.Thread] = {}
_RUNNER_THREADS_LOCK = threading.Lock()


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
        "title": "DUSt3R 多图链路还缺一次完整验收",
        "detail": "前端、参数和远端 runner 都已接通，但还需要用 3 到 5 张图完整验证输出质量与稳定性。",
    },
    {
        "title": "远端取消与清理仍然不够硬",
        "detail": "现在可以本地标记取消并尝试 pkill，但还缺更可靠的远端进程确认和残留目录清理。",
    },
    {
        "title": "MonST3R 真实推理链路正在接入",
        "detail": "服务器权重已经就位，当前目标是跑通官方 demo、拉回 GLB/轨迹/深度等产物，并形成稳定样例。",
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
        "advisor_report": load_advisor_report(job.job_id),
    }


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
            "source_types": SOURCE_TYPE_OPTIONS,
            "advisor": advisor_status(),
        }
    )


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
