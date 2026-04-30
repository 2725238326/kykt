# App Architecture / Performance Optimization Notes

Last updated: 2026-04-26

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
- Added a contract-driven model API layer (`/api/app/state`, `/api/models/catalog`, `/api/models/{model}/contract`, `/api/models/{model}/validate-create`, `/api/jobs/{job_id}/contract`) so the frontend can consume backend model capabilities instead of hard-coding runner rules.
- Added paginated job listing support on `GET /api/jobs` with `limit`, `offset`, `status`, `model`, `source_type`, `sample_id`, `search`, and `sort` query parameters. Existing clients can keep reading `jobs` and `summary`; new clients should use `pageInfo`.
- Added contract-driven artifact indexing on job detail payloads and `GET /api/jobs/{job_id}/artifacts`, grouping returned files by role (`pointcloud`, `scene`, `trajectory`, `frame_preview`, `confidence`, `metadata`, `log`, etc.) and exposing `primaryArtifacts` for inspection-first result views.
- Added `GET /api/jobs/{job_id}/inspection` as a detail-view packet that combines the job payload, model contract, artifact index, log digest, manual evaluation digest, Advisor report state, attention items, and recommended next actions.
- Added Advisor provider diagnostics and schema-oriented OpenAI-compatible integration endpoints (`/api/advisor/providers`, `/api/advisor/diagnostics`, `/api/advisor/test`) with camelCase response aliases for frontend consumption.

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
- Started splitting `client/src/App.tsx` by moving API/bootstrap defaults, parameter presets, and Create parameter option tables into `client/src/appConfig.ts`.
- Moved shared display/formatting helpers into `client/src/displayHelpers.ts` and deployment readiness helpers into `client/src/deploymentHelpers.ts`.
- Moved Sample Matrix report/evidence helpers into `client/src/sampleMatrixHelpers.ts`, file staging helpers into `client/src/fileHelpers.ts`, and Create/job workflow helpers into `client/src/workflowHelpers.ts`.
- Moved shared UI primitives into `client/src/uiPrimitives.tsx`, model roadmap into `client/src/ModelRoadmapPanel.tsx`, Sample Matrix into `client/src/SampleMatrixPanel.tsx`, manual evaluation into `client/src/EvaluationPanel.tsx`, and job result summary rendering into `client/src/SummaryPanel.tsx`.
- Moved advisor rendering into `client/src/AdvisorWorkbench.tsx` and shared job inspector/log/output helpers into `client/src/jobInspectorHelpers.tsx`.
- Moved the Jobs inspector/detail surface into `client/src/JobDetail.tsx`, reducing `client/src/App.tsx` to the workspace shell, data flow, and remaining Create/System panels.
- Added job bundle export via `GET /api/jobs/{job_id}/bundle`, with a Jobs inspector export action.
- Updated Sample Matrix backend mapping so manifest `seed_job_id` values populate matrix cells without mutating historical job records.
- Fast3R runner now handles the local Fast3R loader API and sm75 attention fallback; platform E2E succeeded as job `20260425-113002`.
- Spann3R platform E2E succeeded as job `20260425-113227` using MonST3R frame previews.

### Desktop Shell

- Tauri health checks now hit `/api/health`, which is cheaper and less likely to become slow as job history grows.

## Next Architecture Tasks

1. Extract the remaining Create/System workspace sections from `client/src/App.tsx`, then move shared polling/bootstrap state into focused data hooks.
2. Update the frontend Jobs workbench to consume `pageInfo` and query parameters from `GET /api/jobs`.
3. Unify query / polling state so jobs, samples, and deployment views share more reusable loading/error logic.
4. Update result/detail UI to consume `GET /api/jobs/{job_id}/inspection` as the primary Job Detail data source.
5. Continue tightening report/evaluation/Advisor evidence contracts.
6. Move remaining model-specific UI decisions to backend model contracts.
