# Dream Question Log

Last updated: 2026-05-01

## Round 1

Question: What kind of teacher-facing surprise should the research aim for?

Answer:

- Architecture-level innovation is the priority.
- Cross-modal / new-technology fusion is allowed.
- The work should eventually support paper writing, demo presentation, and KYKT app integration.
- Engineering cost should be light to medium unless the idea clearly deserves more.
- Final naming can wait.

## Round 2

Question: What should the research engine support?

Answer:

- The workspace should be created under `E:\kykt` as `Dream`.
- The desired output form is mixed: architecture story, demo, proposal, and KYKT integration.
- Memory/state, model composition, reasoning/test-time compute, and continual/lifelong learning should all remain open.
- GitHub and paper mining should be fused and adaptive.
- KYKT should eventually contain research idea, fast reproduction, prototype, paper workbench, and long-term work-management areas.
- Evidence standards should be chosen case-by-case using a ladder from mock demo to quantitative metrics.
- Research rules should exist both in the project and later as a reusable Codex skill.

## Next Round Candidate

## Round 3

Question: What should Dream optimize for first: breadth of idea discovery, depth of one architecture route, or a teacher-facing demo path?

Answer:

- Choose a balanced two-track plan.
- Establish a strong research paradigm first, because later results will be weak without it.
- Important decisions should be discussed with the user rather than silently committed.

Decision:

```text
Dream Phase 1 = Breadth Map + Minimal Demo.
```

## Next Round Candidate

## Round 4

Question: Should Dream now arrange a comprehensive research route survey and planning step for the discussed content?

Answer:

- Yes. This step is critical.
- Build the research paradigm and planning scheme first.
- Continue to discuss important decision points with the user.

Decision:

```text
Create Phase 1 research route survey plan.
```

Output:

- `archive/PHASE1_RESEARCH_PLAN.md`

## Round 5

Question: Should Dream start model reproduction now?

Answer:

- No.
- Deploy the research process first.

Decision:

```text
Phase 1.5 = Research Workflow Deployment.
```

Output:

- `paradigm/RESEARCH_WORKFLOW.md`
- `paradigm/RESEARCH_DATA_MODEL.md`
- `WORKFLOW_STATUS.md`
- registries, templates, cycle log, and planned-only experiment file

## Round 6

Question: Did the prior work miss anything or make weak judgments, and should we create a reusable master prompt for agents?

Answer:

- Yes, audit the prior work.
- Create a master prompt that is updated as the project progresses.
- The prompt should force agents to use the Dream markdown files, future skills, and workflow rather than proceeding ad hoc.

Decision:

```text
Create AGENT_MASTER_PROMPT.md as the canonical entry prompt.
Keep archive/MASTER_RESEARCH_PROMPT_DRAFT.md as historical only.
```

Output:

- `AGENT_MASTER_PROMPT.md`
- `decisions/DEC-20260501-006-task-audit-and-master-prompt.md`

## Next Round Candidate

The next useful question is:

```text
Should the next workflow lane be backend research pipeline contract, deeper prompt/rules refinement, planned-only smoke-test details, or a second Dream3R thesis research cycle?
```

Historical correction after user feedback, later superseded by Round 8:

```text
Backend/research pipeline is prioritized over frontend when system integration is needed. Frontend is downstream only.
```

## Round 7

Question: Who should own KYKT frontend design tasks going forward?

Answer:

- Frontend design / implementation tasks should still be handled by another agent, specifically Gemini CLI or a designated frontend implementation agent.
- Dream / Codex should only emphasize, arrange, and maintain the frontend design prompt and task constraints.

Decision:

```text
Gemini CLI owns KYKT frontend implementation by default.
Dream / Codex owns frontend handoff prompts, constraints, sequencing, and acceptance criteria.
```

Output:

- `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md`
- `templates/frontend_design_handoff.md`
- `decisions/DEC-20260501-007-frontend-agent-boundary.md`

## Round 8

Question: Is the app/backend track the mainline, or only a support layer?

Answer:

- The app is only one part.
- The mainline is still researching new content, new mechanisms, and new directions.
- Backend-first should only mean backend before frontend when app integration becomes necessary.

Decision:

```text
Dream is research-content-first.
Backend, app, and frontend are supporting layers.
```

Output:

- `paradigm/RESEARCH_CONTENT_ROADMAP.md`
- `decisions/DEC-20260501-010-research-mainline-correction.md`

## Round 9

Question: Should Dream continue as a linear branch-selection process, or as a nonlinear graph-like research structure?

Answer:

- Many mentioned, unmentioned, or undiscovered modules, papers, techniques, and compositions may become useful at different stages.
- A nonlinear graph or higher-dimensional structure is more suitable than prematurely betting on one branch.
- If writing a top-conference paper, start from field-level problem formulation and mechanism graph, not a single model name.

Decision:

```text
Dream should use a failure-mode / mechanism / composition / evidence graph.
Fill branch comparison before deepening any single thesis branch.
```

Output:

- `planning/RESEARCH_GRAPH_AND_PAPER_START.md`
- `planning/BRANCH_COMPARISON_MATRIX.md`
- updated `AGENT_MASTER_PROMPT.md` short invocation
