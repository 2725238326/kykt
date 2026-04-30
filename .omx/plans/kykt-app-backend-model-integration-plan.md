# KYKT App Backend/API/Model Integration Plan

Last updated: 2026-04-30

## Requirements Summary

The product direction is desktop App first, not website first. The architecture remains:

```text
Tauri desktop shell
  -> React renderer
  -> local FastAPI backend on 127.0.0.1:8765
  -> system ssh/scp
  -> remote model runners on /hdd3/kykt26
```

The backend should become the App capability layer: model catalog, run contracts, parameter schemas, job creation, runner dispatch, result summaries, development lane promotion, and blocked-model explanations should be owned by backend APIs. The frontend should render contracts and call actions, not encode model-specific behavior beyond UI layout.

Grounding facts:

- Current app architecture is documented in `Coding/4.06/vision_ui/APP_ARCHITECTURE_OPTIMIZATION.md` and `Coding/4.06/vision_ui/README.md`.
- `model_registry.py` owns formal model metadata through `ModelSpec`, `MODEL_REGISTRY`, `MODEL_CATALOG`, source types, and dynamic local catalog draft support.
- `app.py` owns request validation, bootstrap/sample/development/job APIs, and still contains model parameter building and source validation.
- `ssh_runner.py` owns remote preparation, runner upload, per-model dispatch, remote status polling, downloads, cancellation, and result summary generation.
- `job_store.py` owns local `JobRecord` persistence under `local_jobs/<job_id>/`.
- `development_store.py` owns persisted Development Lane records and promotion metadata under `local_jobs/development_manifest.json`.
- Current runner files are `dust3r_runner.py`, `mast3r_runner.py`, `monst3r_runner.py`, `spann3r_runner.py`, and `fast3r_runner.py`.

## Architecture Decision

### Decision

Introduce a backend-owned model integration contract layer before adding more UI behavior or new runnable models.

The next backend work should create explicit contracts for:

1. Model catalog and deployment status.
2. Create-job parameter schema.
3. Runner dispatch command contract.
4. Result artifact groups and primary artifacts.
5. Development Lane to JobRecord linkage.
6. Promotion from prototype to catalog draft, then formal runnable model.

### Drivers

- New models should not require editing three or four unrelated switch maps.
- The App should show truthful model state: runnable, catalog-only, blocked, promotion draft, validated.
- Frontend agents should receive stable API contracts and should not infer model behavior.
- Local JSON persistence remains mandatory; no DB migration.
- Remote execution remains through system `ssh/scp`; no inbound remote service.

### Alternatives Considered

1. Keep current pattern: add model branches in `model_registry.py`, `app.py`, `ssh_runner.py`, and runner scripts each time.
   - Pros: fastest for one more model.
   - Cons: keeps model behavior scattered; high chance of stale UI labels and dispatch mismatch.

2. Move all model config to one JSON file immediately.
   - Pros: flexible and less code editing for metadata.
   - Cons: too risky before runner command and result contracts are normalized; dynamic Python callables still need code.

3. Recommended: add typed backend contract modules first, then migrate existing models one by one.
   - Pros: incremental, testable, compatible with existing app and local JSON.
   - Cons: requires a short refactor before Align3R/CUT3R integration.

## Target Backend Shape

### New/Refined Modules

1. `model_registry.py`
   - Keep static curated registry.
   - Add richer fields only when stable:
     - `deployment_key`
     - `runner_file`
     - `runner_kind`
     - `default_params`
     - `param_schema_key`
     - `result_contract_key`
     - `minimum_inputs`
     - `blocked_reason`
   - Keep `local_model_registry.json` as draft/promotion overlay, not source of truth for formal runnable models.

2. New `model_contracts.py`
   - Own typed definitions for:
     - `CreateParamSchema`
     - `RunnerDispatchSpec`
     - `ResultContract`
     - `ArtifactRole`
   - Export helpers:
     - `params_for_model(model, payload)`
     - `minimum_input_count(model, source_type)`
     - `artifact_contract_for(model)`
     - `runner_spec_for(model)`

3. New `runner_dispatch.py` or refactor inside `ssh_runner.py`
   - Replace hard-coded dispatch maps with registry-driven specs.
   - Existing model-specific command functions can remain initially, but routing should come from one contract lookup.

4. `development_store.py`
   - Extend item metadata/linkage:
     - `linkedJobIds`
     - `smokeJobId`
     - `promotion.registryEntry`
     - `promotion.validationErrors`
   - Keep camelCase JSON at API boundary.

5. `job_store.py`
   - Add optional `development_item_id` to `JobRecord`, or store linkage in `DevelopmentItem.metadata.linkedJobIds` if avoiding schema churn.
   - Prefer adding `development_item_id` once API compatibility is checked.

### API Contract Changes

Backend endpoints to add or stabilize:

```text
GET  /api/app/state
GET  /api/models/catalog
GET  /api/models/{model}/contract
GET  /api/models/{model}/deployment
POST /api/models/{model}/validate-create
POST /api/development/lanes/{id}/smoke-test
POST /api/development/lanes/{id}/promote
GET  /api/jobs/{job_id}/contract
```

Recommended response ownership:

- `/api/app/state`: single cheap App bootstrap, replacing frontend need to compose multiple first-load calls over time.
- `/api/models/catalog`: full active/deferred/runnable/blocked/promotion-draft model catalog.
- `/api/models/{model}/contract`: source types, params schema, runner status, result contract, minimum input requirements.
- `/api/development/lanes/{id}/smoke-test`: backend creates a linked JobRecord for a prototype when enough metadata/input sample exists.
- `/api/jobs/{job_id}/contract`: job-specific result contract and recommended inspection order.

Do not require frontend to know whether a model is `video_sequence`, `fast3r_collection`, or `research_catalog` beyond what the API returns.

## Implementation Steps

### Phase 1: Backend Contract Extraction

1. Create `model_contracts.py`.
   - Move `_build_job_params()` logic out of `app.py`.
   - Move `_minimum_input_count()` logic out of `app.py`.
   - Define param schemas for current families:
     - `image_collection`
     - `video_sequence`
     - `spann3r_sequence`
     - `fast3r_collection`
     - `research_catalog`
   - Acceptance: existing `/api/jobs` creation produces identical params for DUSt3R/MASt3R/MonST3R/Spann3R/Fast3R.

2. Add `GET /api/models/{model}/contract`.
   - Include:
     - `model`
     - `runnable`
     - `sourceTypes`
     - `paramSchema`
     - `defaultParams`
     - `minimumInputs`
     - `runnerStatus`
     - `launchBlocker`
     - `resultContract`
   - Acceptance: every model in `/api/bootstrap.model_catalog` has a contract endpoint returning 200 or a clear 404 only for unknown IDs.

3. Update `/api/bootstrap` to remain backward-compatible.
   - Keep existing `models`, `model_catalog`, `source_types`, `advisor`, `summary`.
   - Internally source model catalog through `get_model_catalog_options()`.
   - Acceptance: current frontend build still passes.

### Phase 2: Runner Dispatch Normalization

4. Add registry-backed runner specs.
   - Move runner file map from `ssh_runner._upload_runner()` into a contract/registry helper.
   - Move dispatch map from `ssh_runner.run_remote_job()` into a helper returning one `RunnerDispatchSpec`.
   - Keep existing command builder functions for now.
   - Acceptance: smoke dispatch still chooses the same runner for current five runnable models.

5. Normalize result download policy.
   - Replace `MODEL_RESULT_BUNDLE_MODELS = {"monst3r", "spann3r", "fast3r"}` with result contract fields:
     - `downloadMode: "required_files" | "remote_tree_bundle"`
     - `requiredFiles`
     - `optionalFiles`
   - Acceptance: MASt3R still downloads required `matches.png`/`pointcloud.ply`; MonST3R/Spann3R/Fast3R still bundle remote tree.

6. Normalize result summary generation.
   - Move model-specific summary role logic into `ResultContract`.
   - Keep existing summary renderer, but feed it contract-derived roles.
   - Acceptance: existing completed jobs load summaries; new job summaries include primary artifacts and artifact groups.

### Phase 3: Development Lane Integration

7. Add Development Lane to JobRecord linkage.
   - Add optional `development_item_id` to `JobRecord`, or keep linkage in `DevelopmentItem.metadata.linkedJobIds`.
   - Add API support for linking existing job IDs to a lane.
   - Acceptance: a lane can list linked job IDs and the jobs API can expose the lane relation without breaking older jobs.

8. Add `POST /api/development/lanes/{id}/smoke-test`.
   - Backend validates lane metadata:
     - target model exists.
     - sample/input source exists or a sample ID is provided.
     - model is runnable.
     - param defaults can be built.
   - Backend creates a draft JobRecord linked to the lane.
   - Later phase may dispatch automatically; first version should create but not auto-run unless explicit.
   - Acceptance: lane status can move to `smoke_ready`; created job appears in `/api/jobs`.

9. Tighten promotion flow.
   - Existing promotion writes `local_model_registry.json`; keep this for draft catalog entries.
   - Add validation result payload:
     - `ok`
     - `validationErrors`
     - `registryEntry`
     - `nextBackendSteps`
   - Acceptance: missing runner/env paths return descriptive 400; successful promotion appears in `/api/bootstrap.model_catalog` as non-runnable draft.

### Phase 4: New Model Onboarding Path

10. Define the onboarding checklist as backend data.
   - Add a `model_onboarding` section per catalog item or a separate JSON store:
     - official repo
     - remote repo path
     - env
     - weights
     - smoke command
     - known blocker
     - expected artifacts
   - Acceptance: Align3R and CUT3R blocked state can be rendered from backend data, not hand-written frontend text.

11. Integrate Align3R only after contract extraction.
   - First add catalog contract and blocked deployment contract.
   - Only mark runnable after:
     - remote smoke works.
     - `align3r_runner.py` writes `status.json`, `scene_meta.json`, logs, and result summary-compatible artifacts.
   - Acceptance: before runner readiness, Create cannot launch Align3R and error explains `curope`/CUDA blocker.

12. Integrate CUT3R similarly.
   - Keep blocked until RoPE/`curope` path is resolved.
   - Do not let frontend show it as runnable until backend contract says runnable.

## Acceptance Criteria

- Existing app still builds with `npm run build`.
- Backend imports compile with `.venv\Scripts\python.exe -m py_compile app.py model_registry.py job_store.py ssh_runner.py development_store.py`.
- `/api/bootstrap`, `/api/jobs`, `/api/samples`, `/api/development/lanes` remain backward compatible.
- Every catalog model has a backend-readable contract.
- Adding a new model requires editing one registry/contract path plus adding a runner script, not scattered frontend assumptions.
- Prototype promotion can create a catalog draft without making the model runnable.
- Smoke-test lane action can create a linked JobRecord without requiring frontend-specific logic.
- Blocked models produce stable machine-readable blocker fields.

## Risks and Mitigations

- Risk: Refactor breaks current runnable models.
  - Mitigation: migrate one model family at a time; compare generated params before/after for current jobs.

- Risk: Contracts become too abstract before Align3R/CUT3R are understood.
  - Mitigation: keep model-specific command builders while moving only routing and metadata first.

- Risk: Frontend agent starts encoding model rules again.
  - Mitigation: frontend prompt must explicitly forbid hard-coded model behavior beyond rendering API-provided schemas.

- Risk: Local JSON manifests diverge from static registry.
  - Mitigation: static registry remains curated source of truth; local registry is only promotion draft overlay.

## Verification Steps

Run after each backend phase:

```powershell
cd E:\kykt\Coding\4.06\vision_ui
.\.venv\Scripts\python.exe -m py_compile app.py model_registry.py job_store.py ssh_runner.py development_store.py
cd client
npm run build
```

API smoke checks:

```text
GET /api/health
GET /api/bootstrap
GET /api/models/catalog
GET /api/models/mast3r/contract
GET /api/models/align3r/contract
GET /api/development/lanes
POST /api/development/lanes/{id}/promote
```

Runner smoke checks:

- Do not rerun heavy remote jobs by default.
- Use existing jobs first:
  - Fast3R: `20260425-113002`
  - Spann3R: `20260425-113227`
  - MASt3R: `20260420-222729`
  - MonST3R: `20260420-222928`
- Only run new remote jobs after contract extraction passes local API checks.

## Frontend-Agent Prompt

Use this prompt for a separate frontend agent. Do not give it backend implementation ownership.

```text
You are working on the KYKT Vision desktop App renderer, not a website. Do not redesign the product or create marketing/landing UI. Treat React as the Tauri desktop renderer.

Scope:
- Frontend only.
- Do not modify backend Python files.
- Do not hard-code model-specific runnable/blocked/parameter/result behavior in the frontend.
- Consume backend contracts from:
  - GET /api/bootstrap
  - GET /api/models/catalog
  - GET /api/models/{model}/contract
  - GET /api/development/lanes
  - POST /api/development/lanes/{id}/smoke-test
  - POST /api/development/lanes/{id}/promote

Goal:
Update the desktop renderer so model selection, parameter controls, blocked-model messaging, Development Lane smoke-test actions, and promotion feedback are driven by backend-provided contracts.

Rules:
- Preserve the current Workbench Light desktop style.
- Do not create a website-like page.
- Keep the Overview command center compact and operational.
- When API fields are missing, degrade gracefully and show backend error detail.
- Frontend should render paramSchema/defaultParams from the backend instead of maintaining model-family switches.
- Frontend should render resultContract/primaryArtifacts from the backend instead of inferring artifact roles.
- Do not add new model strings in frontend logic unless only used as display fallback.

Acceptance:
- npm run build passes.
- Existing job creation still works for current runnable models.
- Catalog-only models remain visible but disabled for creation with backend-provided launchBlocker.
- Development Lane panel can create smoke-test jobs when backend allows it and display validation errors when backend blocks it.
- Promotion result displays backend registryEntry and nextBackendSteps without assuming success means runnable.
```

## Recommended Work Order

1. Backend contract extraction.
2. Backend model catalog/contract endpoints.
3. Backend runner dispatch normalization.
4. Development Lane smoke-test/job linkage.
5. Backend promotion response tightening.
6. Frontend agent consumes new contracts.
7. Align3R/CUT3R integration resumes only after the contract path is stable.

