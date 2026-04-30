# KYKT Vision App UI Optimization Prompt

You are the frontend agent for KYKT Vision. Do not change Python backend files. Your task is to redesign the desktop app UI for actual daily use, not to make a landing page or visual demo.

## Context

KYKT Vision is a local desktop workbench for 3D vision model experiments. Users need to:

- create reconstruction jobs from images, frame sequences, or videos;
- choose runnable and catalog-only models from backend contracts;
- inspect job progress, failures, logs, and returned artifacts;
- compare model outputs across shared samples;
- configure AI Advisor providers and read structured evaluation output;
- track Development Lane items from paper/prototype to runner promotion.

The current page is hard to use because too many unrelated surfaces compete for attention. The redesign should make the app feel like a dense, predictable desktop tool.

## Design References To Learn From

- Atlassian/Jira navigation: consistent left navigation, fewer top-level destinations, fast access to search/create/work queues.
- Microsoft Fluent/Windows app patterns: desktop shell, navigation view, command bars scoped to the current view.
- Material design data states: clear empty, loading, error, and no-result states for tables and lists.
- General enterprise app principle: prioritize scanning, filtering, repeated actions, and operational clarity over decorative layout.

## Hard Requirements

1. Do not build a marketing home page.
2. Do not use oversized hero sections, decorative cards, or vague dashboard blocks.
3. Use the backend as the source of truth:
   - `GET /api/app/state`
   - `GET /api/jobs?limit=&offset=&status=&model=&source_type=&sample_id=&search=&sort=`
   - `GET /api/models/catalog`
   - `GET /api/models/{model}/contract`
   - `POST /api/models/{model}/validate-create`
   - `GET /api/jobs/{jobId}/artifacts`
   - `GET /api/advisor/providers`
   - `GET /api/advisor/diagnostics`
4. Do not hard-code model-specific input rules, parameter schemas, result file roles, or provider capabilities in the frontend.
5. Keep the app responsive at desktop sizes first. Mobile is not a priority.
6. Keep UI density high but readable. Avoid nested cards and decorative sections.

## Target Information Architecture

Use a stable left sidebar with these primary workspaces:

1. Queue
   - The default screen.
   - Shows all jobs as a paginated, filterable work queue.
   - Columns: status, model, sample, source, created time, progress, primary artifact, action.
   - Use `pageInfo` from `/api/jobs`.

2. Create
   - A focused job creation workspace.
   - Model picker reads `modelCatalog`.
   - Parameter form reads `modelContracts[model].paramSchema`.
   - Disable catalog-only models and show `launchBlocker`.
   - Validate file count through `/api/models/{model}/validate-create`.

3. Inspect
   - Deep detail for one selected job.
   - Use `artifactIndex.groups` and `artifactIndex.primaryArtifacts`.
   - Layout: left job facts, center artifact groups/result preview, right logs/evaluation/advisor.
   - Do not infer file roles from filename in frontend.

4. Samples
   - Comparison matrix across sample/model combinations.
   - Prioritize evidence readiness, score, model status, and missing runs.

5. Development
   - Development Lane list with priority/status/next action/blockers.
   - Focus on prototype-to-runner flow.

6. System
   - Deployment readiness, Advisor provider config, diagnostics.
   - Keep it utilitarian and compact.

## Interaction Model

- Global search filters jobs and samples.
- The main command bar should change by workspace:
  - Queue: New Job, Refresh, Filter, Export.
  - Create: Validate, Create, Clear.
  - Inspect: Run, Retry, Cancel, Export Bundle, Generate Advisor Report.
  - Samples: Filter, Export Report.
  - Development: New Item, Promote.
  - System: Refresh Diagnostics, Test Advisor.
- Use segmented controls for status/model filters where the set is small.
- Use tables/lists for operational data, not card grids.
- Use drawers or split panes for details instead of modal-heavy navigation.

## Visual Direction

- Build a work-focused desktop app, closer to Jira/Linear/Windows Admin Center than to a SaaS landing page.
- Sidebar + command bar + main content.
- Muted neutral base, restrained accent color, high contrast status chips.
- Small, consistent typography. No viewport-scaled font sizes.
- Tables should have sticky headers, compact rows, visible selected state, and clear empty/error states.
- Use icons in commands where useful, but never hide critical actions behind icon-only buttons without tooltips.

## Backend Contract Mapping

- `AppState.modelContracts` drives Create and artifact/result behavior.
- `JobPayload.artifactIndex.groups` drives output grouping.
- `JobPayload.artifactIndex.primaryArtifacts` drives the first inspection path.
- `/api/jobs` `pageInfo` drives pagination.
- Advisor provider UI must read `/api/advisor/providers` and `/api/advisor/diagnostics`.

## Acceptance Criteria

1. The first screen is a useful Queue, not a decorative overview.
2. User can create a job without reading explanatory text.
3. Catalog-only models are visible but clearly disabled with a concrete blocker.
4. Job detail shows primary artifacts first and logs second.
5. Large job history remains usable through backend pagination.
6. Advisor configuration shows provider, structured output mode, key status, and test result.
7. Existing backend endpoints continue working; no Python files are modified.

