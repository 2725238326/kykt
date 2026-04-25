# Task 424 - Codex Long-Flow Work Order

Last updated: 2026-04-25

## 1. Purpose

This file is the working order for Codex on `E:/kykt/Coding/4.06/vision_ui`.

The goal is not a one-off patch. The goal is to keep moving this project toward a stable local desktop workbench for 3R / visual-geometry model execution, comparison, inspection, deployment tracking, and report output.

Codex should treat this file as a live checklist and update the status boxes as work progresses.

---

## 2. Product Positioning

This project is:

- a local desktop client
- a local FastAPI backend
- an SSH/SCP-driven remote model execution workbench
- a multi-model experiment and inspection tool

This project is not:

- a landing page
- a marketing dashboard
- a mobile-first interface
- a single-model demo page

Current design direction:

- `Workbench Light`
- dense desktop layout
- status-first hierarchy
- comparison-first workflows
- long-session comfort

Primary style source of truth:

- `E:/kykt/Coding/4.06/vision_ui/DESIGN.md`

---

## 3. Mandatory Reading Order

Before starting a new round of work, Codex must quickly read these files in order:

1. `E:/kykt/Coding/4.06/vision_ui/README.md`
2. `E:/kykt/Coding/4.06/vision_ui/DESIGN.md`
3. `E:/kykt/Coding/4.06/vision_ui/ACTIVE_MODEL_INTEGRATION_PLAN.md`
4. `E:/kykt/Coding/4.06/vision_ui/APP_ARCHITECTURE_OPTIMIZATION.md`
5. `E:/kykt/Coding/4.06/vision_ui/MODEL_DEPLOYMENT_STATUS.md`
6. `E:/kykt/Coding/4.06/vision_ui/THREER_MODEL_ROADMAP.md`
7. `E:/kykt/Coding/4.06/vision_ui/CLIENT_REBUILD.md`
8. `E:/kykt/Coding/4.06/vision_ui/SERVER_PREPARATION.md`
9. `E:/kykt/Coding/4.06/vision_ui/MODEL_DOWNLOAD_UPLOAD_PLAN.md`
10. `E:/kykt/Coding/4.06/vision_ui/MONST3R_MAINLINE_PLAN.md`

If a round only touches one narrow area, Codex may skim the unrelated documents, but `DESIGN.md`, `model_registry.py`, `app.py`, and `client/src/App.tsx` should always be treated as core context.

---

## 4. Core Code Entry Points

Frontend:

- `E:/kykt/Coding/4.06/vision_ui/client/src/App.tsx`
- `E:/kykt/Coding/4.06/vision_ui/client/src/appConfig.ts`
- `E:/kykt/Coding/4.06/vision_ui/client/src/displayHelpers.ts`
- `E:/kykt/Coding/4.06/vision_ui/client/src/deploymentHelpers.ts`
- `E:/kykt/Coding/4.06/vision_ui/client/src/styles.css`
- `E:/kykt/Coding/4.06/vision_ui/client/src/types.ts`

Backend:

- `E:/kykt/Coding/4.06/vision_ui/app.py`
- `E:/kykt/Coding/4.06/vision_ui/model_registry.py`
- `E:/kykt/Coding/4.06/vision_ui/job_store.py`
- `E:/kykt/Coding/4.06/vision_ui/ssh_runner.py`

Runner and deployment context:

- `E:/kykt/Coding/4.06/vision_ui/runners/`
- `E:/kykt/Coding/4.06/vision_ui/tools/`

---

## 5. Non-Negotiable Current Facts

### 5.1 Theme and UI direction

- The app must remain `Workbench Light`.
- Do not switch it back to dark mode.
- Keep the desktop workbench feel.
- Avoid landing-page styling, oversized decorative sections, or toy-like controls.

### 5.2 Model input-type truth

The frontend and backend must stay aligned with the registry:

- `DUSt3R` -> `images`
- `MASt3R` -> `images`
- `MonST3R` -> `video / frames`
- `Spann3R` -> `images / frames`
- `Fast3R` -> `images / frames`
- `Align3R` -> `video / frames`
- `CUT3R` -> `video / frames / images`

Do not reintroduce the false assumption that every non-MonST3R model is image-only.

### 5.3 Current create-vs-catalog distinction

The UI must distinguish between:

- runnable models that can be created now
- catalog/research/deployment models that are visible but not creatable yet

Do not hide catalog-only models from the product model story.

### 5.4 Current file boundaries

Primary allowed edit targets:

- `E:/kykt/Coding/4.06/vision_ui/client/src/App.tsx`
- `E:/kykt/Coding/4.06/vision_ui/client/src/styles.css`
- `E:/kykt/Coding/4.06/vision_ui/DESIGN.md`

Secondary targets when needed:

- `E:/kykt/Coding/4.06/vision_ui/client/src/types.ts`
- `E:/kykt/Coding/4.06/vision_ui/app.py`
- `E:/kykt/Coding/4.06/vision_ui/model_registry.py`
- `E:/kykt/Coding/4.06/vision_ui/ssh_runner.py`

Never modify `.omx` files as part of product work.

---

## 6. Long-Flow Objectives

Codex should keep pushing these three lanes together instead of treating them as unrelated:

### A. Workbench UI lane

Build a coherent light desktop workbench across:

- `Overview`
- `Create`
- `Jobs`
- `Sample Matrix`
- `Advisor`
- `System`

Focus:

- scan speed
- batch operation flow
- comparison ergonomics
- inspector rhythm
- consistency of model semantics

### B. Backend model-management lane

Make model behavior derive from a shared contract:

- model label
- family
- source types
- param family
- runner status
- active track
- runnable
- deployment/readiness state

Reduce hardcoded per-model branches where a family/registry-based route is possible.

### C. Remote deployment/use lane

Turn remote model management into product-visible state, not private memory:

- env exists or not
- remote directory exists or not
- required files exist or not
- checkpoints exist or not
- known fallback mode exists or not
- runnable/creatable or blocked
- next action

---

## 7. Execution Loop Per Round

Each round of work should follow this loop:

1. Read the required markdown context.
2. Inspect the relevant code paths.
3. Output a very short execution summary.
4. Pick 2 to 4 tightly related tasks from this file.
5. Implement them directly.
6. Validate.
7. Update the checklist status in this file.
8. Report completed work, changed files, validation results, and next best tasks.

Do not stop at proposal level unless blocked by missing information or broken local state.

---

## 8. Validation Requirements

Every round must run:

Frontend build:

```bash
cd E:/kykt/Coding/4.06/vision_ui/client && npm run build
```

Backend syntax check:

```bash
python -m py_compile E:/kykt/Coding/4.06/vision_ui/app.py E:/kykt/Coding/4.06/vision_ui/job_store.py E:/kykt/Coding/4.06/vision_ui/model_registry.py E:/kykt/Coding/4.06/vision_ui/ssh_runner.py
```

If the round materially changes the desktop UX, also run:

```bash
cd E:/kykt/Coding/4.06/vision_ui/client && npm run tauri build
```

---

## 9. Prohibited Moves

- Do not revert to dark theme.
- Do not turn pages into marketing layouts.
- Do not hardcode model assumptions that contradict `model_registry.py`.
- Do not add UI-only text that explains the app instead of supporting actual workflow.
- Do not break current create/dispatch/result chains for visual polish.
- Do not touch `.omx`.
- Do not do unrelated refactors.

---

## 10. Active Task Checklist

### Phase 0 - Context and alignment

- [x] Re-read the mandatory markdown files before each substantial new round.
- [x] Keep `DESIGN.md`, `App.tsx`, `model_registry.py`, and `app.py` aligned.
- [x] Remove stale frontend assumptions that contradict backend registry or deployment status.

### Phase 1 - Create page as a true model-control surface

- [x] Make `Create` use `model_catalog` instead of a legacy hardcoded model picker.
- [x] Separate runnable models from catalog-only models in the Create workflow.
- [x] Show supported input types, family, and runner state for the selected model.
- [x] Stop treating every non-MonST3R model as plain image-only.
- [x] Give `Fast3R` a model-appropriate parameter panel.
- [x] Give `Spann3R` an honest fixed-parameter explanation instead of fake MonST3R-style controls.
- [x] Make Create parameter routing fully derive from `param_family`, not mostly from model-name branches.
- [x] Decide whether `Align3R` / `CUT3R` should become creatable now or remain catalog-only with explicit blockers.
- [x] Add clear per-model launch blockers when a model is visible but not currently creatable.

### Phase 2 - Unified model semantics across all pages

- [x] Ensure `Jobs`, `Sample Matrix`, and `System` all expose the same model semantics:
  - `label`
  - `family`
  - `source_types`
  - `runner_status`
  - `runnable`
- [x] Remove page-local wording that contradicts registry/deployment truth.
- [x] Make model status chips read consistently between Create, Jobs, Matrix, and System.

### Phase 3 - Jobs workbench improvements

- [x] Add stronger batch operations in `Jobs`.
- [x] Add better filtered-selection handoff behavior.
- [x] Add log keyword highlighting, not just filtering.
- [x] Add a stronger attention-first view for failed/cancelled jobs.
- [x] Improve the right-side inspector rhythm for summary/evidence/logs/evaluation.
- [x] Add keyboard-centric navigation (`/`, `J/K`, arrow support in search).

### Phase 4 - Sample Matrix as a real compare workspace

- [x] Add sort/filter and bulk ID operations.
- [x] Add unassigned-job pool visibility.
- [x] Add clearer row/column compare hierarchy for samples vs models.
- [x] Map manifest `seed_job_id` jobs into Sample Matrix cells for existing MASt3R / MonST3R baseline jobs.
- [x] Surface runner/deployment constraints inside matrix context where useful.
- [x] Add stronger score/evidence digest for quick compare.
- [x] Add exportable compare/report path for selected sample/model subsets.

### Phase 5 - System page as model-management console

- [x] Add deployment readiness matrix.
- [x] Add a clearer separation between:
  - local service state
  - remote deployment state
  - model creatability
  - model research/catalog presence
- [x] Surface fallback modes such as slow path / attention fallback / missing `curope`.
- [x] Show next-action guidance per blocked model, not just global summaries.
- [x] Tie `System` model state more directly to `Create` availability.

### Phase 6 - Backend model-management cleanup

- [ ] Audit `app.py` for model-name hardcoding that should move to registry/family logic.
- [ ] Audit `ssh_runner.py` and runners for inconsistent output contracts.
- [ ] Make deployment/readiness state machine more explicit and reusable.
- [ ] Ensure bootstrap and samples APIs expose enough model metadata for frontend decisions.

### Phase 7 - Remote model lifecycle and usage

- [ ] Define a clearer lifecycle for:
  - code directory
  - env
  - weights/checkpoints
  - smoke status
  - runner contract
  - frontend availability
- [ ] Make remote checker output directly useful to frontend and deployment panel.
- [ ] Reduce manual operator knowledge required to know whether a model is ready.

### Phase 8 - Delivery and reporting

- [x] Add job bundle export.
- [x] Add sample/model comparison report export.
- [ ] Improve advisor/evaluation/report contract consistency.
- [ ] Redesign AI/advisor evaluation around an evidence contract instead of generic diagnosis copy.
- [x] Reduce noisy on-page guidance copy and downgrade current AI/advisor UI to an auxiliary draft path.
- [x] Make the app better for final inspection and handoff, not only internal testing.
- [x] Put usability improvements into the 3R roadmap as a first-class route, not only as visual polish.

### Phase 9 - Current frontend cleanup priorities

- [x] Restore and keep `Workbench Light` as the only active theme direction.
- [x] Collapse `client/src/styles.css` to a single light-theme foundation plus layout refinement.
- [x] Keep the Jobs page as the main desktop workbench pattern.
- [x] Move shared API/default bootstrap/model parameter config out of `client/src/App.tsx`.
- [x] Move shared display/formatting and deployment readiness helpers out of `client/src/App.tsx`.
- [ ] Continue splitting `client/src/App.tsx` into workspace-sized components, hooks, and helper modules.
- [x] Run platform end-to-end Fast3R validation (`20260425-113002`).
- [x] Run platform end-to-end Spann3R validation (`20260425-113227`).
- [ ] Inspect Fast3R / Spann3R output quality through the desktop client and select better comparison samples.

---

## 11. Output Format for Codex Each Round

After every round, report:

1. What was completed.
2. Which files changed.
3. What logic was unified or corrected.
4. Whether validation passed.
5. If packaging ran:
   - latest exe path
   - latest msi path
   - timestamps
6. Which 2 to 3 tasks should be tackled next.

Keep the report concise and engineering-focused.
