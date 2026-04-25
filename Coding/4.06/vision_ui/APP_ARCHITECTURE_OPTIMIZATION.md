# App Architecture / Performance Optimization Notes

Last updated: 2026-04-25

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
- Added a catalog-driven Create workspace that distinguishes runnable models from catalog-only models.
- Unified model semantics across Create, Jobs, Sample Matrix, and System using `model_catalog` / `/api/samples`.
- Added stronger Jobs workbench behavior: batch actions, filtered-selection handoff, attention-first failure handling, and keyboard-centric navigation.
- Added Sample Matrix compare tooling: sorting, filtering, bulk ID copy, report export, and locate-job handoff.
- Added a system-page deployment console backed by `/api/deployment/status`.
- Consolidated `client/src/styles.css` onto a single Workbench Light foundation plus layout refinement instead of stacked dark/light override passes.

### Desktop Shell

- Tauri health checks now hit `/api/health`, which is cheaper and less likely to become slow as job history grows.

## Next Architecture Tasks

1. Split `client/src/App.tsx` into workspace containers and data hooks.
2. Add pagination or windowing for very large local job histories.
3. Add optional job-bundle export for selected jobs.
4. Unify query / polling state so jobs, samples, and deployment views share more reusable loading/error logic.
