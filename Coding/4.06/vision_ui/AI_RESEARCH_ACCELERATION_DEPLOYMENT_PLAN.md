# KYKT Vision AI + Research Acceleration Deployment Plan

Last updated: 2026-04-30

## 1. Direction

KYKT Vision should not become a generic chatbot wrapper. The AI layer must become a local research co-pilot that improves the actual 3D vision workflow:

1. reduce failed-run diagnosis time;
2. make model comparison more evidence-based;
3. turn jobs, artifacts, logs, and evaluations into next experimental actions;
4. help promote paper/prototype work into stable runners;
5. preserve local-first operation and avoid mandatory cloud infrastructure.

The backend is the product boundary. The frontend should render backend state and AI task outputs; it should not hard-code model rules, prompt logic, file role inference, or provider behavior.

## 2. Current Baseline

Already in place:

- local FastAPI backend and Tauri desktop shell;
- JSON-backed job store and development lane store;
- contract-driven model catalog and artifact indexing;
- job inspection packet at `GET /api/jobs/{job_id}/inspection`;
- OpenAI-compatible Advisor provider support with schema-oriented outputs;
- provider diagnostics and test endpoints;
- local advisor trace file under `local_jobs/_advisor/advisor_calls.jsonl`.

Main weakness:

- `advisor.py` is doing too many jobs at once: provider config, prompt assembly, schema validation, request execution, diagnostics, and trace writing.
- AI is still mostly report generation, not an operational decision layer.
- There is no first-class backend concept of AI tasks, evaluation datasets, benchmark runs, or recommendation history.
- Observability is local-only and useful, but not yet structured enough to compare prompts/models/providers over time.

## 3. External Patterns To Adopt

Use these patterns, not necessarily these libraries immediately:

- OpenAI Structured Outputs: make every AI task schema-first so the backend can validate answers and reject malformed output.
- OpenAI Agents SDK tracing model: treat each AI task as a trace with spans for context building, provider call, validation, and post-processing.
- LiteLLM gateway pattern: keep KYKT compatible with OpenAI-style APIs, while allowing optional routing, budgets, retries, provider fallback, and rate limits through a local gateway later.
- Langfuse observability pattern: traces, sessions, prompt versions, datasets, experiments, manual labels, and LLM-as-judge scores are useful, but should remain optional export targets.
- Ragas evaluation loop: move from one-off "looks okay" checks toward repeatable datasets, metrics, experiments, and regression comparison.
- Pydantic AI output pattern: model outputs should be validated against typed backend objects, with explicit retries or failure states when validation fails.

Sources:

- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- LiteLLM docs: https://docs.litellm.ai/
- Langfuse docs: https://langfuse.com/docs
- Ragas docs: https://docs.ragas.io/
- Pydantic AI output docs: https://pydantic.dev/docs/ai/core-concepts/output/

## 4. Target Backend Architecture

```text
app.py
  -> ai_tasks.py              # task APIs: diagnose, compare, recommend, promote-readiness
  -> ai_gateway.py            # provider invocation, retry, timeout, budget, response parsing
  -> ai_schemas.py            # Pydantic request/response contracts
  -> ai_context.py            # builds evidence packets from jobs, artifacts, samples, dev lanes
  -> ai_trace_store.py        # local JSONL traces, scores, cost, latency, validation failures
  -> advisor.py               # compatibility wrapper during migration

job_store.py
  -> evidence source for jobs/logs/artifacts/evaluations

model_contracts.py
  -> evidence source for model input/output contracts

development_store.py
  -> evidence source for paper/prototype promotion state
```

The key change is to make "AI Advisor" a set of typed backend tasks instead of one broad report endpoint.

## 5. AI Task Layer

Add these backend task types first:

### 5.1 Job Failure Diagnosis

Endpoint:

```text
POST /api/ai/tasks/job-diagnosis
```

Input:

- `jobId`
- optional `focus`: `failure`, `quality`, `runtime`, `artifact_missing`

Output:

- `summary`
- `rootCauseCandidates[]`
- `evidence[]` with log/artifact/job references
- `recommendedActions[]`
- `confidence`
- `requiresHumanCheck`

Purpose:

- make failed jobs actionable in under one minute;
- avoid frontend parsing logs;
- convert failure patterns into reusable backend diagnostics.

### 5.2 Next Experiment Recommendation

Endpoint:

```text
POST /api/ai/tasks/next-run
```

Input:

- `jobId` or `sampleId`
- optional comparison context

Output:

- `recommendedRuns[]`
- `parameterChanges[]`
- `sampleChanges[]`
- `stopConditions[]`
- `why`

Purpose:

- push research forward by recommending the smallest useful next experiment.

### 5.3 Model Comparison Summary

Endpoint:

```text
POST /api/ai/tasks/compare-models
```

Input:

- `sampleId`
- `jobIds[]`
- optional metric filters

Output:

- `winnerByUseCase[]`
- `tradeoffs[]`
- `missingEvidence[]`
- `reportMarkdown`

Purpose:

- generate evidence-backed comparison summaries from existing job artifacts, not generic model descriptions.

### 5.4 Development Promotion Readiness

Endpoint:

```text
POST /api/ai/tasks/promotion-readiness
```

Input:

- `developmentItemId`

Output:

- `ready`
- `blockingIssues[]`
- `runnerContractDraft`
- `testPlan`
- `registryDraftNotes`

Purpose:

- connect Development Lane items to model registry promotion without inventing registry entries prematurely.

### 5.5 Research Report Draft

Endpoint:

```text
POST /api/ai/tasks/research-report
```

Input:

- `sampleIds[]`
- `modelIds[]`
- optional `dateRange`

Output:

- `abstract`
- `method`
- `results`
- `limitations`
- `nextWork`
- `evidenceIndex[]`

Purpose:

- make KYKT generate usable research notes and comparison reports from actual runs.

## 6. AI Gateway Requirements

The first implementation should stay lightweight and local:

- no mandatory database;
- no mandatory Docker service;
- no mandatory Langfuse/LiteLLM dependency;
- use JSON files under `settings/` and `local_jobs/_ai/`;
- provider calls remain OpenAI-compatible over HTTP.

Gateway responsibilities:

- provider config normalization;
- request timeout and retry policy;
- structured output mode selection;
- schema validation;
- token/cost/latency capture when provider returns usage;
- redaction before trace writing;
- optional response cache keyed by task type + evidence hash + schema version;
- clear provider error mapping for UI display.

Suggested files:

```text
settings/ai_gateway.json
local_jobs/_ai/task_traces.jsonl
local_jobs/_ai/task_cache/
local_jobs/_ai/eval_scores.jsonl
```

Do not add LiteLLM as a required dependency yet. Keep the backend compatible with a LiteLLM proxy by allowing `baseUrl` and `apiKey` in config. Later, users who want routing, budgets, fallback, and spend tracking can point KYKT at `http://127.0.0.1:4000`.

## 7. Research Workflow Upgrade

The app should guide a real research loop:

```text
Sample design
  -> model run matrix
  -> job inspection
  -> manual/automatic evaluation
  -> AI diagnosis and next-run recommendation
  -> benchmark report
  -> development lane promotion or deferral
```

Backend concepts to add:

### 7.1 Benchmark Manifest

Create `benchmark_store.py` with local JSON persistence:

```text
local_jobs/benchmark_manifest.json
```

Entities:

- `BenchmarkSet`: sample group, target models, target params, acceptance criteria;
- `BenchmarkRun`: generated jobs, status, missing cells, completed cells;
- `BenchmarkScore`: human score, heuristic score, AI score, evidence references.

### 7.2 Evidence Packet

Add a reusable `EvidencePacket` builder:

- job facts;
- logs tail;
- artifact index;
- model contract;
- manual evaluations;
- previous AI task traces;
- development lane status;
- benchmark context.

Every AI task must receive an evidence packet and must return evidence references. If no evidence supports a claim, the output must mark it as an assumption.

### 7.3 Research Metrics

Track practical metrics first:

- time-to-first-smoke;
- failure diagnosis turnaround;
- missing benchmark cells;
- successful runner promotions;
- per-sample best model;
- quality score trend per model;
- model runtime and VRAM notes where available.

Do not over-focus on abstract LLM evaluation. The research value comes from making 3D vision experiments repeatable and comparable.

## 8. Observability And Evaluation

Local-first trace schema:

```json
{
  "traceId": "ai_...",
  "taskType": "job_diagnosis",
  "schemaVersion": "2026-04-30",
  "provider": "openai",
  "model": "gpt-...",
  "startedAt": "...",
  "endedAt": "...",
  "latencyMs": 1234,
  "usage": {"inputTokens": 0, "outputTokens": 0, "costUsd": null},
  "evidenceHash": "...",
  "validation": {"ok": true, "retries": 0},
  "redactions": [],
  "jobId": "...",
  "sampleId": null
}
```

Optional future export:

- Langfuse export for traces, prompt versions, datasets, experiments, and manual labels;
- LiteLLM proxy for provider routing, budgets, fallback, rate limiting, and provider-level logs;
- Ragas-style offline evaluation for AI task prompts, especially diagnosis correctness and report faithfulness.

Policy:

- KYKT must work without these external services.
- External observability should be opt-in.
- Prompt, context, and artifact content should be redacted by default when it may include local paths, private data, or images.

## 9. Deployment Scheme

### 9.1 Local Desktop Default

Default user path:

```text
Run kykt_vision_client.exe
  -> Tauri starts local backend
  -> backend reads settings/*.json
  -> jobs and AI traces persist under local_jobs/
```

Direct executable for testing:

```text
E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\kykt_vision_client.exe
```

### 9.2 Provider Configuration

Keep `settings/advisor.json` compatible, but introduce `settings/ai_gateway.json`:

```json
{
  "defaultProvider": "openai",
  "defaultModel": "gpt-5.1",
  "baseUrl": "https://api.openai.com/v1",
  "timeoutSeconds": 60,
  "maxRetries": 1,
  "dailyBudgetUsd": null,
  "traceMode": "local",
  "sendImagesByDefault": false,
  "cacheEnabled": true
}
```

Environment variables should override secrets:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `LITELLM_API_KEY`

### 9.3 Optional LiteLLM Mode

When enabled:

```json
{
  "defaultProvider": "litellm",
  "baseUrl": "http://127.0.0.1:4000",
  "defaultModel": "gpt-5.1"
}
```

KYKT should still call an OpenAI-compatible endpoint. LiteLLM remains outside the bundled app unless explicitly deployed.

### 9.4 Versioned App Builds

After each visible app update:

1. run backend checks;
2. rebuild the direct Tauri executable;
3. tell the user to open `client\src-tauri\target\release\kykt_vision_client.exe`;
4. do not prioritize installer generation unless packaging/distribution is requested.

For planning-only or docs-only changes, no executable rebuild is needed.

## 10. Implementation Roadmap

### Phase A: AI Service Split

Deliverables:

- `ai_schemas.py`
- `ai_trace_store.py`
- `ai_gateway.py`
- `ai_context.py`
- compatibility wrapper from `advisor.py`

Acceptance:

- existing Advisor endpoints keep working;
- each AI call writes a structured local trace;
- invalid structured output returns a descriptive backend error;
- provider diagnostics report gateway config, key status, and last task failure.

### Phase B: First Operational AI Tasks

Deliverables:

- `POST /api/ai/tasks/job-diagnosis`
- `POST /api/ai/tasks/next-run`
- `GET /api/ai/tasks/{traceId}`
- `GET /api/ai/tasks?jobId=&taskType=&limit=`

Acceptance:

- a failed job can produce a structured diagnosis;
- a completed job can produce a next-run recommendation;
- outputs include evidence references and confidence;
- traces can be listed from the backend.

### Phase C: Benchmark Store

Deliverables:

- `benchmark_store.py`
- `GET /api/benchmarks`
- `POST /api/benchmarks`
- `POST /api/benchmarks/{id}/plan-runs`
- `GET /api/benchmarks/{id}/matrix`

Acceptance:

- backend can represent target sample/model matrices;
- missing runs are explicit;
- benchmark scores can mix human, heuristic, and AI values.

### Phase D: Research Report + Promotion Readiness

Deliverables:

- `POST /api/ai/tasks/compare-models`
- `POST /api/ai/tasks/research-report`
- `POST /api/ai/tasks/promotion-readiness`
- promotion task can feed Development Lane and local registry draft.

Acceptance:

- report contains evidence index;
- promotion readiness identifies missing runner contract items;
- merged development items create an auditable trace.

### Phase E: Optional Observability Export

Deliverables:

- local trace export format;
- optional Langfuse export adapter;
- optional LiteLLM deployment notes;
- prompt/eval dataset export.

Acceptance:

- KYKT remains fully usable without external services;
- exports do not include raw images or private local paths unless explicitly enabled;
- prompt regressions can be evaluated against saved trace/evidence datasets.

## 11. Immediate Next Backend Slice

Start with Phase A and the smallest useful Phase B endpoint:

1. create `ai_schemas.py` for task contracts;
2. create `ai_trace_store.py` with atomic JSONL writes and trace listing;
3. create `ai_gateway.py` using the current advisor provider logic but with a cleaner API;
4. create `ai_context.py` for `build_job_evidence_packet(job_id)`;
5. add `POST /api/ai/tasks/job-diagnosis`;
6. update the frontend prompt with new endpoint contracts;
7. run backend syntax checks;
8. rebuild the direct exe only if app runtime code changed and the UI can expose the new capability.
