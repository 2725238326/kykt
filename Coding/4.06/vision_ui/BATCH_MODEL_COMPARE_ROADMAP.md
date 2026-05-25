# Batch Model Compare Roadmap

Last updated: 2026-05-25

## Product Direction

The app should grow from a model launcher into a 3R evaluation and delivery workbench:

```text
one input sample
-> selected model set
-> batch job creation
-> controlled remote dispatch
-> result collection
-> visual comparison board
-> scoring and diagnosis
-> report / asset bundle export
```

This keeps the useful aggregation layer, but makes the business function explicit: help the user decide which 3R model works best for a specific input and produce evidence that can be shown or handed off.

## Functional Modules

### 1. Input Readiness Assistant

Purpose: reduce failed model runs before GPU time is spent.

Minimum features:

- detect source type: images, frames, video
- count images / frames and flag too few inputs for selected models
- show model compatibility before task creation
- later: blur, exposure, duplicate-frame, texture, and parallax checks

Acceptance:

- A user can see why a selected model cannot run before upload dispatch.
- Batch creation rejects invalid model/input combinations with per-model messages.

### 2. Batch Model Compare

Purpose: run the same sample through multiple runnable models.

Implemented backend base:

- `POST /api/compare/batches`
  - multipart form upload
  - `models`: JSON array or comma-separated model ids
  - `source_type`
  - optional `sample_id`
  - optional `params` for shared params
  - optional `model_params` object keyed by model id
  - optional `auto_dispatch`
- all created jobs share one `sample_id`
- returns created jobs and a compare packet

Remaining:

- frontend multi-model picker
- run mode controls: create-only / dispatch-now
- dispatch concurrency policy in UI
- saved batch history

Acceptance:

- Upload once, choose two or more compatible runnable models, create jobs with a shared `sample_id`.
- `/api/samples` and `/api/compare/samples/{sample_id}` show those jobs under the same sample.

### 3. Dispatch Queue

Purpose: avoid overloading remote GPU / SSH by blindly launching every model at once.

Current state:

- Existing remote dispatch launches one thread per job.
- Batch endpoint can dispatch immediately, but it does not yet enforce a real queue.

Next implementation:

- add batch-level queue state
- configurable max concurrent remote jobs, default 1
- stop / pause / resume batch
- batch progress derived from child jobs

Acceptance:

- Batch run does not launch more than the configured concurrency.
- Cancelling a batch attempts to cancel all running child jobs and marks pending children clearly.

### 4. Visual Comparison Board

Purpose: generate a side-by-side evidence surface rather than a text-only report.

Implemented backend base:

- `GET /api/compare/samples/{sample_id}` returns:
  - model cells
  - statuses
  - score snapshots
  - primary artifacts
  - visual artifact candidates
  - Markdown report text

Frontend/display work:

- render one row per model
- show first primary visual prominently
- show secondary artifact strip
- support image, video, pointcloud, and GLB placeholders/previews
- support “download comparison board” once screenshot/rendering exists

Later backend/export work:

- server-side HTML report
- generated contact sheet PNG/PDF
- zipped compare package

Acceptance:

- A finished sample can be opened as a visual board with one column or row per model.
- The board makes missing/failed/finished cells visually obvious.

### 5. Quality Scoring And Diagnosis

Purpose: convert raw output into useful decisions.

Current state:

- Manual evaluation exists per job.
- Sample Matrix can summarize score snapshots.

Next implementation:

- required rubric by sample type
- model-specific checklist:
  - MASt3R: matches + pointcloud
  - MonST3R: GLB + trajectory + frames + dynamic masks
  - Spann3R: pointcloud + transforms
  - Fast3R: pointcloud + cameras + confidence
  - Align3R / CUT3R: after runners land
- Advisor task: compare models within one `sample_id`

Acceptance:

- The report can state “best current model for this sample” with evidence.
- Missing primary artifacts are treated as review blockers.

### 6. Report And Asset Delivery

Purpose: make the software useful after the run completes.

Current state:

- Per-job bundle export exists.
- Sample Matrix Markdown report exists in the frontend.
- New backend compare endpoint returns Markdown report text.

Next implementation:

- backend compare report endpoint with Markdown already exists:
  - `GET /api/compare/samples/{sample_id}/report`
- add zipped compare bundle:
  - report
  - selected visuals
  - each child job bundle link or copied core artifacts
- later: HTML/PDF export

Acceptance:

- A user can export one folder/zip that explains the comparison and contains the core evidence files.

### 7. New Model Onboarding Agent

Purpose: turn “quickly deploying new 3R models” into a real product capability.

Modules:

- repo / paper intake
- environment checklist
- checkpoint checklist
- official demo smoke tracking
- runner draft checklist
- output contract draft
- platform smoke report
- promotion into runnable catalog

Acceptance:

- A new model can move through states:
  - research candidate
  - env ready
  - official smoke passed
  - runner pending
  - platform smoke passed
  - matrix ready

## Updated Task Route

### P0: Make Batch Compare Usable

1. Backend batch creation API.
2. Backend sample compare packet API.
3. Frontend batch create flow.
4. Frontend visual comparison board.
5. Manual test with `mast3r + fast3r` or `spann3r + fast3r` on image inputs.

### P1: Make It Reliable

1. Batch dispatch queue with max concurrency.
2. Batch cancel / retry controls.
3. Batch status persistence.
4. Compare bundle export.

### P2: Make It Insightful

1. Compare-level Advisor task.
2. Model recommendation from evidence.
3. Input readiness checks.
4. Auto-generated teaching / presentation wording.

### P3: Make It Extensible

1. Model Onboarding Agent.
2. Runner contract validator.
3. Smoke report generator.
4. Promotion workflow from catalog-only to runnable.

## Current Backend Progress

Completed on 2026-05-25:

- Added batch compare creation endpoint.
- Added sample compare packet endpoint.
- Added sample compare Markdown report endpoint.
- Fixed API job creation so the frontend `params` JSON form field is honored by the backend.

Still open:

- frontend multi-model creation UI
- frontend visual board
- queued dispatch
- compare bundle export
- real screenshot/contact-sheet generation
