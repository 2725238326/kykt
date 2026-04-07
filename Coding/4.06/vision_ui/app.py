from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from job_store import (
    ROOT,
    clear_job_runtime,
    create_job,
    duplicate_job,
    get_log_snippets,
    iter_input_items,
    list_jobs,
    load_job,
    save_inputs,
    update_job,
)
from ssh_runner import ServerConfig, cancel_remote_job, run_remote_job


app = FastAPI(title="KYKT Vision UI", version="0.3.0")

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.globals["asset_version"] = "20260407-1110"

(ROOT / "static").mkdir(parents=True, exist_ok=True)
(ROOT / "local_jobs").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/local_jobs", StaticFiles(directory=str(ROOT / "local_jobs")), name="local_jobs")


PHASE_FLOW = [
    ("local_prepared", "本地任务已就绪", "本地任务记录和输入缓存已经准备好。", 4, 8),
    ("preparing_remote", "准备服务器目录", "正在创建远端任务目录和任务文件。", 8, 15),
    ("uploading_inputs", "上传输入文件", "正在把输入文件和任务清单发送到服务器。", 15, 25),
    ("running_remote_matches", "运行 DUSt3R 重建", "正在构建图像配对、执行推理、全局对齐并生成匹配图。", 25, 70),
    ("running_remote_pointcloud", "导出点云", "正在导出点云并完成远端输出文件。", 70, 90),
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

ACTIVE_PHASE_CODES = [code for code, *_ in PHASE_FLOW[:6]]
PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


def status_label(status: str | None) -> str:
    if not status:
        return "未知"
    return STATUS_LABELS.get(status, status)


templates.env.globals["status_label"] = status_label


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


@app.get("/")
async def index(request: Request):
    jobs = list_jobs(limit=50)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": jobs,
            "server": ServerConfig(),
            "models": [
                ("dust3r", "DUSt3R（图像集）"),
                ("monst3r", "MonST3R（视频或帧序列）"),
            ],
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
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="没有上传输入文件。")

    params = {}
    if model == "dust3r":
        params = _dust3r_params(image_size, scene_graph, niter, lr, batch_size, max_points, match_viz_count)

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
async def dispatch_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    if len(job.input_files) < 2 and job.model == "dust3r":
        raise HTTPException(status_code=400, detail="DUSt3R 至少需要两张输入图片。")

    clear_job_runtime(job_id)
    background_tasks.add_task(run_remote_job, job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    if len(job.input_files) < 2 and job.model == "dust3r":
        raise HTTPException(status_code=400, detail="DUSt3R 至少需要两张输入图片。")

    clear_job_runtime(job_id)
    background_tasks.add_task(run_remote_job, job_id)
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
    return JSONResponse({"jobs": payload})


@app.get("/api/jobs/{job_id}")
async def job_detail_api(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务 {job_id}。") from exc

    return JSONResponse(_job_payload(job))
