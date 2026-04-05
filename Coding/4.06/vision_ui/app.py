from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from job_store import ROOT, create_job, list_jobs, load_job, save_inputs
from ssh_runner import ServerConfig, describe_next_remote_steps, run_remote_job


app = FastAPI(title="KYKT Vision UI", version="0.1.0")

templates = Jinja2Templates(directory=str(ROOT / "templates"))

(ROOT / "static").mkdir(parents=True, exist_ok=True)
(ROOT / "local_jobs").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/local_jobs", StaticFiles(directory=str(ROOT / "local_jobs")), name="local_jobs")


@app.get("/")
async def index(request: Request):
    jobs = list_jobs()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": jobs,
            "server": ServerConfig(),
            "models": [
                ("dust3r", "DUSt3R (two or more images)"),
                ("monst3r", "MonST3R (video / frame sequence)"),
            ],
        },
    )


@app.post("/jobs")
async def create_job_view(
    model: str = Form(...),
    source_type: str = Form(...),
    notes: str = Form(""),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No input files were uploaded.")

    job = create_job(model=model, source_type=source_type, notes=notes)
    uploaded = []
    for upload in files:
        uploaded.append((upload.filename, await upload.read()))
    save_inputs(job, uploaded)

    return RedirectResponse(url=f"/jobs/{job.job_id}", status_code=303)


@app.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    previews = []
    for rel_path in job.input_files:
        suffix = Path(rel_path).suffix.lower()
        previews.append(
            {
                "relative_path": rel_path,
                "url": "/" + rel_path.replace("\\", "/"),
                "is_image": suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
            }
        )

    outputs = []
    for rel_path in job.output_files:
        suffix = Path(rel_path).suffix.lower()
        outputs.append(
            {
                "relative_path": rel_path,
                "url": "/" + rel_path.replace("\\", "/"),
                "is_image": suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
                "is_pointcloud": suffix == ".ply",
                "is_log": suffix == ".log",
            }
        )

    next_steps = describe_next_remote_steps(job.model)
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "job": job,
            "previews": previews,
            "outputs": outputs,
            "next_steps": next_steps,
        },
    )


@app.post("/jobs/{job_id}/dispatch")
async def dispatch_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.") from exc

    background_tasks.add_task(run_remote_job, job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)
