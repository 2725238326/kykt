from __future__ import annotations

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
)
from ssh_runner import ServerConfig, run_remote_job


app = FastAPI(title="KYKT Vision UI", version="0.3.0")

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.globals["asset_version"] = "20260406-1328"

(ROOT / "static").mkdir(parents=True, exist_ok=True)
(ROOT / "local_jobs").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/local_jobs", StaticFiles(directory=str(ROOT / "local_jobs")), name="local_jobs")


PHASE_FLOW = [
    ("local_prepared", "Local Job Ready", "The local record and cached inputs are ready.", 4, 8),
    ("preparing_remote", "Preparing Remote Space", "Creating remote folders and job files.", 8, 15),
    ("uploading_inputs", "Uploading Inputs", "Sending files and the job manifest to the server.", 15, 25),
    ("running_remote_matches", "Running DUSt3R Matches", "Building matches, alignment, and match visualization.", 25, 70),
    ("running_remote_pointcloud", "Exporting Point Cloud", "Exporting the point cloud and finishing remote outputs.", 70, 90),
    ("downloading_results", "Downloading Results", "Copying outputs and logs back to the local cache.", 90, 98),
    ("finished", "Finished", "The job completed successfully.", 100, 100),
    ("failed", "Failed", "The job stopped with an error. Check logs and retry.", 0, 0),
]

ACTIVE_PHASE_CODES = [code for code, *_ in PHASE_FLOW[:6]]
PROGRESS_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


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
    current_index = ACTIVE_PHASE_CODES.index(phase) if phase in ACTIVE_PHASE_CODES else len(ACTIVE_PHASE_CODES)
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
) -> dict:
    return {
        "image_size": min(max(int(image_size), 224), 1024),
        "scene_graph": scene_graph.strip() or "complete",
        "niter": min(max(int(niter), 0), 1000),
        "lr": max(float(lr), 0.0),
        "batch_size": min(max(int(batch_size), 1), 8),
        "max_points": min(max(int(max_points), 1000), 2_000_000),
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
                ("dust3r", "DUSt3R (image set)"),
                ("monst3r", "MonST3R (video or frame sequence)"),
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
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No input files were uploaded.")

    params = {}
    if model == "dust3r":
        params = _dust3r_params(image_size, scene_graph, niter, lr, batch_size, max_points)

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
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

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
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    if len(job.input_files) < 2 and job.model == "dust3r":
        raise HTTPException(status_code=400, detail="DUSt3R needs at least two input images.")

    clear_job_runtime(job_id)
    background_tasks.add_task(run_remote_job, job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    if len(job.input_files) < 2 and job.model == "dust3r":
        raise HTTPException(status_code=400, detail="DUSt3R needs at least two input images.")

    clear_job_runtime(job_id)
    background_tasks.add_task(run_remote_job, job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/duplicate")
async def duplicate_job_view(job_id: str):
    try:
        new_job = duplicate_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    return RedirectResponse(url=f"/jobs/{new_job.job_id}", status_code=303)


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
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    return JSONResponse(_job_payload(job))
