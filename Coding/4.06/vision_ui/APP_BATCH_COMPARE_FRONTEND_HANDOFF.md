# Frontend Handoff: Batch Model Compare

Last updated: 2026-05-25

This handoff is for the designated frontend agent. Codex has prepared backend support and product constraints; frontend implementation should follow the existing Workbench Light layout and avoid turning this into a marketing-style page.

## Goal

Add a first usable “one input, multiple models, comparison board” workflow.

The user flow:

```text
Create workspace
-> select Batch Compare mode
-> upload one input set
-> choose compatible models
-> create batch jobs
-> optionally dispatch now
-> open compare board by sample_id
```

## Backend APIs

### Create Batch

`POST /api/compare/batches`

Multipart form fields:

- `models`: JSON array string like `["mast3r","fast3r"]` or comma string like `mast3r,fast3r`
- `source_type`: `images`, `frames`, or `video`
- `sample_id`: optional; backend generates one if blank
- `notes`: optional
- `auto_dispatch`: boolean, default false
- `params`: optional JSON object shared by all models
- `model_params`: optional JSON object keyed by model id
- `files`: one or more uploaded files

Response:

- `sampleId`
- `createdJobs`
- `dispatchResults`
- `compare`

### Read Compare Board Packet

`GET /api/compare/samples/{sample_id}`

Response includes:

- `summary`
- `modelCells`
- `model_cells`
- `reportMarkdown`
- per-model `visuals`, `primaryArtifacts`, `outputs`, and `previews`

### Read Markdown Report

`GET /api/compare/samples/{sample_id}/report`

Returns `text/markdown`.

## UI Requirements

### Create Flow

- Add a segmented control or tab inside Create: `Single Job` / `Batch Compare`.
- Batch mode should reuse existing file staging and model catalog data.
- Show only runnable models by default.
- Disable incompatible model/source combinations before submit.
- Show model status chips using existing helpers.
- Provide `Create only` and `Create + dispatch` actions.

### Compare Board

- Add a route/workspace state for compare board by `sample_id`.
- Display:
  - sample id
  - job count
  - finished/running/attention counts
  - average score when available
  - one model row/card per `modelCell`
- For each model cell:
  - status
  - job id link to inspector
  - primary artifact
  - visual artifact strip
  - score snapshot
  - progress message
- Missing visuals should show a compact placeholder, not an empty blank area.

### Export Actions

- Copy report Markdown.
- Download report Markdown from `/api/compare/samples/{sample_id}/report`.
- Later: call compare bundle endpoint when backend adds it.

## Visual Constraints

- Keep dense operational layout.
- Do not create a hero/landing page.
- Do not put cards inside cards.
- Use existing panel/table/card patterns from `SampleMatrixPanel`.
- Use stable dimensions for model cells and preview tiles.
- Long file names must wrap or truncate without layout shift.
- Use existing model/status helper labels instead of hard-coded display logic where possible.

## Acceptance Criteria

- Uploading one input set and choosing `mast3r,fast3r` creates two jobs with the same `sample_id`.
- The Sample Matrix shows both jobs under the same sample if `sample_id` matches a manifest sample.
- Opening the compare board for that `sample_id` shows one model cell per created job.
- The board remains useful when jobs are still running, failed, or missing visual outputs.
- `npm run build` passes.
