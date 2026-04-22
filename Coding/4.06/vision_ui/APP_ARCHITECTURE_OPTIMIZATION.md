# App Architecture / Performance Optimization Notes

Last updated: 2026-04-21

## Current Architecture

```text
Tauri desktop shell
  -> React client
  -> local FastAPI backend on 127.0.0.1:8765
  -> system ssh/scp
  -> remote model runner on /hdd3/kykt26
```

The app should remain a local desktop workbench. The server should run model jobs only; no inbound server-side service is required.

## Optimizations Completed

### Backend

- Added `GET /api/health` as a lightweight health endpoint.
- Changed the Tauri backend health probe from `/api/bootstrap` to `/api/health`, avoiding repeated dashboard/job/model payload generation for simple liveness checks.
- Added mtime-based caching for `samples_manifest.json`.
- Changed log tail reading in `job_store.get_log_snippets()` from full-file reads to tail-window reads, reducing cost when runner logs grow.
- Updated delivery gaps to match the active 3R model route instead of stale DUSt3R multi-image wording.

### Frontend

- Reduced idle polling pressure:
  - job list polling uses 4s only when jobs are running; otherwise 12s
  - selected job detail uses 4s only while selected job is running; otherwise 15s
  - samples polling uses 60s
- Added model roadmap and sample/evaluation panels backed by `model_catalog` and `/api/samples`.
- Added sample scoring-category overview and seed job hints.
- Added a manual evaluation panel in task detail with persisted scoring fields and notes.
- Added a system-page remote deployment summary panel backed by `/api/deployment/status`.

### Desktop Shell

- Tauri health checks now hit `/api/health`, which is cheaper and less likely to become slow as job history grows.

## Next Architecture Tasks

1. Add `spann3r_runner.py` and `fast3r_runner.py`.
2. Add a model-comparison view grouped by sample id.
3. Add pagination or windowing for very large local job histories.
4. Add optional result bundle export for selected jobs and model-comparison reports.
