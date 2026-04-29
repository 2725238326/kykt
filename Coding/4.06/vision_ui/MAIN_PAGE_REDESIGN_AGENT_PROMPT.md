# Prompt for next agent: KYKT Vision main-page redesign

You are working on the KYKT Vision local desktop client:

```text
E:\kykt\Coding\4.06\vision_ui
```

Your task is to improve the KYKT app main page and product design logic. The new design should make the app feel more like a serious 3R / visual-geometry workbench command center, and it should reserve a clear product lane for future "rapid reproduction / rapid idea implementation" workflows inside the KYKT app.

Do **not** modify `E:\Demo\Idea_Implementer`. That folder is only context for a future separate agent. This task is about the KYKT app UI/design.

## Required reading

Read these files first and treat them as the current source of truth:

1. `E:\kykt\KYKT.md`
2. `E:\kykt\PROJECT_PROGRESS_2026-04-23.md`
3. `E:\kykt\Coding\4.06\vision_ui\DESIGN.md`
4. `E:\kykt\Coding\4.06\vision_ui\ACTIVE_MODEL_INTEGRATION_PLAN.md`
5. `E:\kykt\Coding\4.06\vision_ui\MODEL_DEPLOYMENT_STATUS.md`
6. `E:\kykt\Coding\4.06\vision_ui\APP_ARCHITECTURE_OPTIMIZATION.md`
7. `E:\kykt\Coding\4.06\vision_ui\CLIENT_REBUILD.md`
8. `E:\kykt\Coding\4.06\vision_ui\THREER_MODEL_ROADMAP.md`

Then inspect the current client implementation:

- `E:\kykt\Coding\4.06\vision_ui\client\src\App.tsx`
- `E:\kykt\Coding\4.06\vision_ui\client\src\styles.css`
- `E:\kykt\Coding\4.06\vision_ui\client\src\workflowHelpers.ts`
- `E:\kykt\Coding\4.06\vision_ui\client\src\types.ts`
- extracted panels such as `ModelRoadmapPanel.tsx`, `SampleMatrixPanel.tsx`, `AdvisorWorkbench.tsx`, and `JobDetail.tsx`

## Product context

KYKT Vision is a **local desktop 3R / visual geometry model workbench**, not a landing page and not a generic chatbot.

It currently supports:

- model job creation and dispatch
- remote execution status
- MASt3R / MonST3R / Spann3R / Align3R / Fast3R / CUT3R tracking
- result inspection, logs, artifacts, manual evaluation, and advisor summaries
- sample matrix and model comparison
- deployment/system readiness

The design source of truth is `DESIGN.md`: Workbench Light, dense operational desktop UI, status-first, comparison-friendly, no marketing hero, no decorative dashboard style.

## Design objective

Improve the app's main page / Overview so the first screen behaves like an operations command center for these lanes:

1. **Run models**: create and monitor concrete 3R jobs.
2. **Compare results**: sample matrix, model comparison, output quality.
3. **Check readiness**: deployment, blocked models, system status.
4. **Accelerate development**: future rapid reproduction / rapid idea implementation workflow inside the KYKT app.

The final UI should help the user answer within 2 seconds:

- What is currently running?
- What is blocked?
- What result should be inspected next?
- What model or sample should be compared next?
- What research/prototype lane is next in the development cycle?

## New design lane to add

Add an explicit **Research / Implementation Acceleration** lane to the KYKT app design.

This lane is for future workflows such as:

- fast reproduction of new papers/projects
- fast implementation of rough research ideas
- turning an idea into a scoped prototype plan
- checking whether an idea should become a runner, experiment, report, or deferred research note
- merging validated prototypes back into KYKT's main model/workbench flow

For this task, do not assume a full backend exists. The UI may honestly present this lane as planned / design-ready / placeholder. Do not claim it is functional before it is wired.

Suggested development-cycle model:

```text
Idea intake -> Reproduction plan -> Prototype runner -> Smoke test -> Unified output contract -> Sample Matrix comparison -> Report / merge decision
```

Suggested states:

- `draft`: rough idea captured
- `scoped`: dependencies, inputs, outputs, and success criteria identified
- `reproducing`: upstream repo or paper reproduction underway
- `prototype`: local implementation is being built
- `smoke_ready`: a first non-interactive run can be attempted
- `validated`: output contract and result evidence are good enough to compare
- `merged`: promoted into the main KYKT Vision flow
- `deferred`: parked as research context

Future data shape, if useful:

```ts
type DevelopmentLaneItem = {
  id: string;
  title: string;
  category: "paper_reproduction" | "model_runner" | "prototype" | "evaluation" | "ui_workflow";
  status: "draft" | "scoped" | "reproducing" | "prototype" | "smoke_ready" | "validated" | "merged" | "deferred";
  priority: "P0" | "P1" | "P2" | "P3";
  targetModel?: string;
  nextAction: string;
  blockers: string[];
  mergeTarget?: "runner" | "sample_matrix" | "advisor" | "report" | "deferred_research";
};
```

Do not overbuild persistence unless explicitly asked. Static typed placeholder config is acceptable for this design pass if isolated and easy to replace later.

## Concrete UI work

Improve the Overview around these surfaces:

1. **Focus Strip**
   - local service status
   - running jobs
   - attention jobs
   - advisor readiness
   - active model/deployment blockers
   - next development-cycle action

2. **Development Cycle Panel**
   - compact horizontal or matrix-style stage tracker
   - show active lanes: model integration, result evaluation, and research/prototype acceleration
   - make it clear what is active, blocked, planned, and ready to merge

3. **Research Acceleration Panel**
   - status should be honest: planned/design-ready unless real wiring exists
   - show next actions:
     - define input contract for paper/project reproduction
     - define output contract for prototype artifacts
     - define how validated prototypes merge into `model_registry`, runners, Sample Matrix, Advisor, or reports
   - show 2-4 seed categories:
     - paper reproduction
     - new 3R model runner
     - UI/evaluation workflow prototype
     - research report / experiment design

4. **Roadmap integration**
   - keep active 3R model route visible: MASt3R, MonST3R, Spann3R, Align3R, Fast3R, CUT3R
   - show the acceleration lane as a support mechanism, not as a replacement for current active model integration

5. **Action routing**
   - route to existing pages where possible:
     - create runnable model job -> Create
     - inspect result -> Jobs
     - compare sample/model -> Sample Matrix
     - check blocked env -> System
     - draft evaluation/report -> Advisor
   - for acceleration lane actions, placeholder behavior is acceptable: show a compact note or focus the development-cycle panel.

## Design constraints

Follow `DESIGN.md` strictly:

- Keep Workbench Light.
- Keep desktop workbench density.
- Do not create a hero page.
- Do not add marketing copy.
- Do not use oversized cards or decorative gradients.
- Use panels, compact status rows, matrices, and clear operational grouping.
- Preserve the current information architecture unless there is a strong reason to change it.
- Keep Overview / Create / Jobs / Sample Matrix / System / Advisor concepts coherent.

If you add a new top-level nav item, justify it. A better first pass may be to expose the acceleration lane on Overview and roadmap surfaces instead of adding a full new page immediately.

## Engineering guidance

Keep changes scoped and maintainable.

Preferred structure:

- Extract a new component if Overview grows further, for example:
  - `client/src/DevelopmentCyclePanel.tsx`
  - `client/src/ResearchAccelerationPanel.tsx`
- Keep static data/config near existing app config if no backend endpoint exists yet.
- Avoid making `App.tsx` much larger; it is already being gradually split.
- Reuse existing primitives where possible:
  - `PanelTitle`
  - `StatusBadge`
  - `MiniStat`
  - `SummaryStat`
  - `MessageBanner`
  - `ModelSemanticChips`
- Keep CSS aligned with existing Workbench Light classes.

Do not:

- add large dependencies
- redesign the whole app shell from scratch
- remove existing keyboard flow in Jobs
- break Create / Jobs / Sample Matrix / System / Advisor
- hide blocked model state
- claim the acceleration lane is functional before it is wired

## Acceptance criteria

The result is acceptable if:

- Overview clearly communicates the app as a 3R workbench command center
- the development-cycle lane appears in the main app design
- the design remains compact, operational, and aligned with `DESIGN.md`
- existing app routes still work
- TypeScript builds without errors
- the implementation can later connect to a real backend/job contract without redoing the UI concept

Run at least:

```powershell
cd E:\kykt\Coding\4.06\vision_ui\client
npm run build
```

If there is an existing preferred validation command in the repo, run that too.

## Final response expected

When finished, report:

- files changed
- what changed in the main page / design logic
- what remains placeholder vs functional
- build/test result
- recommended next step for wiring the acceleration lane into backend data or job contracts

