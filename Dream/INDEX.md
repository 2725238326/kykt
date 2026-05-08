# Dream Index

Last updated: 2026-05-08 (cycle 031 DONE: Memory v0.3 local P0 scaffold NEW; ABL-memory-0 passed as fixture/logging validity gate; next recommended = ABL-memory-1 local vector baseline)

Quick navigation for humans and agents. **Read `TASK_SNAPSHOT.md` first** (it is the highest-authority resume pointer; if its status is `in_progress` or `blocked`, do not start new work). Then read `AGENT_MASTER_PROMPT.md` for full operating rules.

## How To Read This Workspace

1. Start with `TASK_SNAPSHOT.md` (highest-authority resume pointer; tells you whether a task is in flight, what subtask is next, and whether forward motion is gated).
2. Then `AGENT_MASTER_PROMPT.md` (entry prompt with mandatory load protocol; `TASK_SNAPSHOT.md` is item 1 of that protocol, the rest follows).
3. Check `WORKFLOW_STATUS.md` for current phase, blocked decisions, and recommended next user decision.
4. Check `RESEARCH_STATE.md` for living state and cycle history.
5. Use this `INDEX.md` to find any specific file by topic.

## Root-Level Files (Always-On Entry Points)

| File | Role |
|---|---|
| `TASK_SNAPSHOT.md` | **Read first.** Highest-authority resume pointer: current task id, subtask board, status (`idle` / `in_progress` / `blocked`), `If interrupted, resume from` block, recent failure modes |
| `README.md` | Workspace overview, purpose, non-negotiables |
| `INDEX.md` | This file; topic-based navigation |
| `AGENT_MASTER_PROMPT.md` | Canonical agent operating prompt + mandatory load protocol (lists `TASK_SNAPSHOT.md` as item 1) |
| `WORKFLOW_STATUS.md` | Current phase, active workstreams, blocked decisions, recommended next user decision |
| `RESEARCH_STATE.md` | Living state log; cycle history; current recommendations |

## Subdirectories By Purpose

### `paradigm/` - How Dream Operates

| File | Role |
|---|---|
| `RESEARCH_PARADIGM.md` | Operating paradigm, two-track plan, evidence ladder, user-discussion gates |
| `RESEARCH_WORKFLOW.md` | Source-to-implementation operational workflow |
| `RESEARCH_DATA_MODEL.md` | Schema for sources, mechanisms, RUs, decisions, experiments |
| `RESEARCH_SKILL_RULES_DRAFT.md` | Evolving rules for a future Codex skill |
| `RESEARCH_CODE_DISCIPLINE.md` | Behavior rules for synthesis and Dream-driven code (Karpathy-adapted + honesty override) |
| `RESEARCH_CONTENT_ROADMAP.md` | Research-content-first roadmap and axes |
| `CROSS_SPEC_SIGNAL_CONTRACT.md` | Formal contract for read-only / handoff signals between finalist specs; v1 covers Critic / Memory / Permanence / Composer |
| `TEACHER_AUDIENCE_PROFILE.md` | Placeholder file for the user to populate; gates D3 (first teacher demo target) |

### `planning/` - Active Research Planning

| File | Role |
|---|---|
| `MULTI_TRACK_RESEARCH_CANVAS.md` | Multi-branch comparison canvas (do not collapse to single thesis prematurely) |
| `RESEARCH_GRAPH_AND_PAPER_START.md` | Failure-mode / mechanism / composition graph; paper scaffold |
| `BRANCH_COMPARISON_MATRIX.md` | Branch-level comparison matrix and score table |
| `BRANCH_SHORTLIST_DECISION_SURFACE.md` | User decision surface for choosing 2-3 finalist branches |
| `ARCHITECTURE_MECHANISM_INTAKE.md` | Branch-neutral intake map for broad architecture and visual mechanisms |
| `ACTION_TAXONOMY_AND_PROXY_METRICS.md` | Compact A1-A8 actions and P1-P8 proxy validation protocols |
| `DREAM3R_THESIS_STRESS_TEST.md` | Stress test for the Dream3R / GEM-3R candidate branch |
| `MINIMAL_DEMO_CANDIDATES.md` | Teacher-demo candidate analysis |
| `WORK_RISK_REGISTER.md` | Consolidated cross-spec risk view across the four finalist specs and the cross-spec contract |
| `L3_PILOT_SELECTION.md` | Cycle 014 L3 downselect: recommends Critic as first pilot and Composer as backup; planning only, not L3 authorization. Cycle 015 acted on this recommendation: DEC-20260505-005 authorized the Critic L3 pilot SCOPE (per-step micro gates still required) |
| `COMPOSER_CAPABILITY_DESCRIPTORS.md` | Cycle 018 S2 deliverable: 7 admitted lightweight 3R experts (MASt3R / Fast3R / Spann3R / CUT3R / MoGe-2 / DepthAnything-V2 / Test3R) with capability descriptor schema (innovation point / input / output / cost / attention regime / adapter / capability_match / failure modes / evidence labels); routing policy sketch; cross-axis summary table; pool admission criteria + drops (VGGT, MapAnything, Kimi-KDA) |
| `NSA_MEMORY_INTEGRATION_MEMO.md` | Cycle 018 S3 deliverable: NSA × C2 Memory integration sketch; three-branch (compressed / selected / sliding) mapped to v0.2 Memory hierarchy A+B; bounded anchor bank K=256 proposed; selection gate driven by Critic confidence + Permanence link; LM-to-3R transfer labeled speculative |
| `MEMORY_V03_DESIGN_STUDY.md` | Cycle 025 direction 3.1 deliverable: code-observed study of CUT3R learned state-token recurrence and Spann3R explicit SpatialMemory; compares both against current Dream3R GRU + vector AnchorBank + NSAThreeBranch C2; proposes C2 v0.3 as latent state tokens + explicit spatial key/value bank + bus-gated writes; markdown research only |
| `MEMORY_V03_P0_PROTOTYPE_PLAN.md` | Cycle 027 deliverable: markdown-only P0 static tensor prototype plan for C2 Memory v0.3. Defines deterministic tensor fixtures, regimes R1-R5, variants V0 vector AnchorBank / V1 spatial key-value bank / V2 state-token recurrence / V3 hybrid bus-gated writes, metrics M1-M8, kill conditions K1-K7, reviewer checklist, and separate execution gate. Engineering plan only; no implementation authorized |
| `MEMORY_V03_ABLATION_REVIEW.md` | Cycle 029 review artifact for SPEC-20260508-002. Verdict: approved for planning use after corrections, not approved for execution. Captures R-029-1..5 corrections: oracle-bus boundary, state-token stale-smooth fail, op-proxy-only cost claim, hard/soft fail rules, narrowed controlled loop/revisit claim |
| `MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md` | Cycle 030 deliverable: future execution authorization template for local static tensor P0 ABL-memory-0..8. Predefines allowed path, forbidden paths/actions, oracle-bus boundary, required outputs, stop gates, result labels, go/no-go rules, and evidence boundary. Template only; no implementation authorized |
| `DINOV3_C1_INTEGRATION_MEMO.md` | Cycle 018 S3 deliverable: DINOv3 × C1 Perceiver integration sketch; DINOv3-S replaces ViT-L (~14x param reduction; ~5x latency speedup; paper-derived); -B fallback documented; frozen-backbone default; heads from scratch |
| `DREAM3R_V02_CODE_STRUCTURE.md` | Cycle 020 S2 deliverable: v0.2 code structure planning. Maps SPEC-20260506-004 v0.2 architecture Delta 1..6 to per-file changes in `code/dream3r/`. Per-file change manifest with MODIFIED (modules.py / model.py / losses.py / smoke_test.py / config.py / train.py) + NEW (memory_anchor_bank.py / nsa_attention.py / composer_experts/ subdir with 7 expert adapters / latency_bench.py) + STABLE (bus.py CR-1..CR-6 carry forward). Explicit review surface subsection per MODIFIED + NEW file per user request "其他agent审阅修改". Server-side deployment surface per F-002. Honest evidence labels: speculative for NSA-related; paper-derived for DINOv3-S; engineering-judgment for expert pool composition; inferred for line-of-code estimates. Sibling artifact to `DREAM3R_V02_IMPLEMENTATION_ROADMAP.md` |
| `DREAM3R_V02_IMPLEMENTATION_ROADMAP.md` | Cycle 020 S3 deliverable: v0.2 implementation roadmap. Breaks `DREAM3R_V02_CODE_STRUCTURE.md` per-file changes into 22 reviewable tasks across 4 tiers — Tier 1 prerequisites (T-v02-A repo skeleton + T-v02-B test harness); Tier 2 per-module (T-v02-C1 DINOv3-S backbone / T-v02-C2-mem-bank / T-v02-C2-nsa / T-v02-C5 Composer routing / T-v02-EXPERT-1..7 per-expert adapters); Tier 3 integration (T-v02-D model wiring / T-v02-E losses update / T-v02-F smoke + latency bench); Tier 4 ABL harnesses (T-v02-ABL-1..9 one per ABL-v02-N from SPEC-005). Per-task scope/inputs/outputs/effort/pre-execution review checklist/post-execution validation checklist/execution gate. Task dependency DAG. ~250-380 engineering-hours total inferred (post-first-task calibration needed). Per-task review checklists are the load-bearing other-agent handoff hook per user request "其他agent审阅修改" |

### `sources/` - Source Mining

| File | Role |
|---|---|
| `FRONTIER_SOURCE_MAP.md` | Verified and pending source map; cycle-tagged source-mining passes |

### `units/` - Research Units, Scoring, Reproduction Readiness

| File | Role |
|---|---|
| `RESEARCH_UNIT_BANK.md` | Structured Dream Research Units (RU-001..) |
| `IDEA_SCOREBOARD.md` | Score table for candidate ideas |
| `REPRODUCTION_READINESS_MATRIX.md` | Repo-level smoke-test and KYKT integration readiness |

### `handoff/` - Collaboration And Frontend

| File | Role |
|---|---|
| `FRONTEND_DESIGN_HANDOFF_PROMPT.md` | Canonical frontend handoff prompt for Gemini CLI |
| `COLLABORATION_ROADMAP.md` | Human-agent collaboration path and near-term deployment sequence |

### `logs/` - Running Logs

| File | Role |
|---|---|
| `QUESTION_LOG.md` | Interview history and next questions |

### `archive/` - Historical / Superseded

| File | Role |
|---|---|
| `PHASE1_RESEARCH_PLAN.md` | Phase 1 plan (historical; Phase 1.5 is current) |
| `PHASE1_EXECUTION_LOG.md` | Phase 1 running log (historical) |
| `PHASE1_DECISION_MEMO.md` | Phase 1 synthesis and gates (historical) |
| `MASTER_RESEARCH_PROMPT_DRAFT.md` | Superseded by `AGENT_MASTER_PROMPT.md` |

### `cycles/` - Per-Cycle Research Logs

Format: `CYCLE-YYYYMMDD-NNN.md`. Newest is the active cycle.

### `decisions/` - Decision Memos

Format: `DEC-YYYYMMDD-NNN-<slug>.md`. Indexed in `registry/decision_registry.md`.

### `code/dream3r/` - Dream3R PyTorch Implementation

Dream3R v0.1 code. Runs on remote server `/hdd3/kykt26/code/dream3r/` (dream3r conda env). Local copy is for editing; deploy via SFTP.

| File | Role |
|---|---|
| `bus.py` | C6 Memory Bus: typed signal namespace + CR-1..CR-6 gates (zero parameters) |
| `modules.py` | C1 Perceiver (ViT backbone + heads), C2 Memory (GRU/Mamba), C3 Permanence (Slot Attention), C4 Critic (TransformerEncoder), C5 Composer (table join) |
| `model.py` | Dream3R main model: wires C1-C6 in bus tick order; preset configs (small / small_vit) |
| `losses.py` | Multi-loss L_total: pointmap + critic P1/P5 + permanence P4 + action entropy |
| `smoke_test.py` | End-to-end validation: forward + backward + bus signals + CR-1 gate + memory carry-over |
| `PLAN.md` | Implementation roadmap (6 phases) with checklist |

### `experiments/` - Experiment Plans

Format: `EXP-YYYYMMDD-NNN-<slug>.md`. Filing here does not mean the experiment was run. Cycle 013 added 4 L3 prerequisites briefs (one per finalist; brief-only, NOT L3 authorization).

| File | Finalist | Role |
|---|---|---|
| `EXP-20260501-001-dust3r-splatt3r-smoke-plan.md` | Phase 1 baseline | First reproducibility plan; planned only, do not run until user confirms |
| `EXP-20260505-001-l3-prerequisites-critic.md` | Geometry Critic (SPEC-20260503-001) | L3 prerequisites brief: repos + checkpoints + GPU/disk/time (`inferred`) + smoke-test path + minimum code change. Cycle 015 SCOPE authorized per DEC-20260505-005; per-step micro gates G_clone / G_install / G_download / G_run / G_log_use still required before any operational step |
| `EXP-20260505-002-l3-prerequisites-memory.md` | Executive Memory (SPEC-20260503-002) | L3 prerequisites brief; same structure |
| `EXP-20260505-003-l3-prerequisites-permanence.md` | Dynamic Object Permanence (SPEC-20260503-003) | L3 prerequisites brief; same structure |
| `EXP-20260505-004-l3-prerequisites-composer.md` | 3R Composer (SPEC-20260504-001) | L3 prerequisites brief; same structure; closure of G2 inventoried but not executed |
| `prototypes/memory_v03_p0/README.md` | C2 Memory v0.3 | Cycle 031 local P0 scaffold. `ABL-memory-0` passed 22/22 fixture/logging validity checks and generated `outputs/` artifacts. Later `ABL-memory-1..8` behavior remains unimplemented and requires new DEC |

### `cases/` - L2 Proxy Case Cards

Format: `CASE-YYYYMMDD-<SPEC>-NNN.md`. One file per (input, finalist spec) pair under `templates/proxy_case_card.md`. Filing a case card does not claim measured performance; the template enforces evidence labels (paper-proven / inferred / demo-observed / code-observed). Cycle 009 populated the first portfolio:

| File | Spec | Role |
|---|---|---|
| `CASE-20260504-CRITIC-01.md` | SPEC-20260503-001 | Static pair (MASt3R upstream); A5 = rerun_local_region; CR-1 not exercised |
| `CASE-20260504-CRITIC-02.md` | SPEC-20260503-001 | Fast3R vs Spann3R; CR-1 reroute_model + Composer agree/veto loop |
| `CASE-20260504-CRITIC-03.md` | SPEC-20260503-001 | MonST3R 48-frame; CR-3 forward-reference read of Memory latent_drift_proxy |
| `CASE-20260505-COMPOSER-01.md` | SPEC-20260504-001 | Static-collection regime; CR-1 closure paired with CRITIC-02 |
| `CASE-20260505-COMPOSER-02.md` | SPEC-20260504-001 | Regime-typed route_regret central thesis card |
| `CASE-20260505-COMPOSER-03.md` | SPEC-20260504-001 | Fast3R vs MASt3R-SfM; v1 -> v2 cost-typed route_regret canonical under v2 (per DEC-20260504-004) |
| `CASE-20260505-COMPOSER-04.md` | SPEC-20260504-001 | KYKT-metadata-derived capability_card grounded to 4 KYKT job inventory (cycle 012; advances G2 inferred -> inferred-with-real-inventory-anchor; G2 NOT closed; first non-paper-derived Composer L2 card) |
| `CASE-20260505-COMPOSER-05.md` | SPEC-20260504-001 | Cycle 014 VGGT capability-card gap addendum; per-card gap, no v2.2 contract revision; G2 NOT closed |
| `CASE-20260504-MEMORY-01.md` | SPEC-20260503-002 | MonST3R 48-frame; primary Memory L2; CR-3 producer (closes cycle-009 CRITIC-03 forward-reference null) |
| `CASE-20260504-MEMORY-02.md` | SPEC-20260503-002 | Spann3R transforms timeline; externalization-of-governance argument |
| `CASE-20260504-MEMORY-03.md` | SPEC-20260503-002 | MASt3R small-N baseline; non-hallucination boundary on static-pair regime |
| `CASE-20260504-PERMANENCE-01.md` | SPEC-20260503-003 | MonST3R 48-frame; primary Permanence L2; CR-2 producer (closes cycle-009 gap G1 with MEMORY-01) |
| `CASE-20260504-PERMANENCE-02.md` | SPEC-20260503-003 | MASt3R static control; mint_object_id rate = 0 (closes PERMANENCE-01 fail_fast c) |
| `CASE-20260504-PERMANENCE-03.md` | SPEC-20260503-003 | Synthetic dynamic identity-validation; closes PERMANENCE-01 fail_fast b |

### `literature/` - Literature Guidance Board

Curated guidance, not inventory. Inventories live in `sources/FRONTIER_SOURCE_MAP.md` and `registry/source_registry.md`.

| File | Role |
|---|---|
| `INDEX.md` | Entry point; usage rules; pointers to inventories so guidance and inventory do not duplicate |
| `SPINE_CRITIC.md` | Required + advanced + skip-with-reason reading for the Critic finalist (SPEC-20260503-001); cross-paper disagreements; spec interface |
| `SPINE_MEMORY.md` | Same structure for the Executive Memory finalist (SPEC-20260503-002) |
| `SPINE_PERMANENCE.md` | Same structure for the Dynamic Object Permanence finalist (SPEC-20260503-003) |
| `SPINE_COMPOSER.md` | Same structure for the Composer finalist (SPEC-20260504-001); MoE routing as cross-domain analog |
| `CRITICAL_NOTES.md` | Running log of "looks like X is X' but actually" insights; deconfusion of commonly-confused mechanisms |
| `PAPER_RELATED_WORK_SKELETON.md` | Cycle 013 upgraded from skeleton to **prose draft** (Sections 1-7 prose anchored to L2 case cards + SRC-* IDs; Sections 8-9 drafted as prose). Filename retained per Surgical Edits |
| `PAPER_PHASE2_BLUEPRINT.md` | Cycle 014 claim-safe paper blueprint; separates current L2 / inferred claims from L3-required claims; not full paper readiness |

### `specs/` - Finalist Mechanism Specs and Architecture Specs

Format: `SPEC-YYYYMMDD-NNN-<slug>.md`. One file per user-approved finalist branch or architecture-level deliverable. Created via `templates/finalist_mechanism_spec.md`. Drafting a spec does not authorize reproduction, training, checkpoint download, or KYKT navigation change.

| File | Role |
|---|---|
| `SPEC-20260503-001-geometry-critic.md` | Geometry Critic / System-2 3R finalist spec (A4 + A5 repair facet; P1 + P5) |
| `SPEC-20260503-002-executive-memory.md` | Executive Memory / State Governance finalist spec (A1 + A2 + A3; P2 + P3) |
| `SPEC-20260503-003-dynamic-object-permanence.md` | Dynamic Object Permanence / 4D Memory finalist spec (A6; P4 + identity_consistency) |
| `SPEC-20260504-001-3r-composer.md` | 3R Composer / Unified Model Ecology finalist spec (A5 routing facet; P5 route_regret + capability_match) |
| `SPEC-20260506-001-dream3r-architecture.md` | Dream3R architecture v0.1: control-graph-as-architecture; hybrid substrate (transformer + SSM + slot + bus); 4 finalist specs synthesized as cores C1-C5 + C6 bus; CR-1..CR-6 as gates; A1-A8 mapped to concrete layers (cycle 016 S2) |
| `SPEC-20260506-002-dream3r-ablation-plan.md` | Dream3R ablation plan v0.1: 10 ablations in 3 tiers; falsification table per architectural claim; benchmark categories B1-B6; dependency graph (cycle 016 S3) |
| `SPEC-20260506-003-dream3r-comparator-map.md` | Dream3R comparator map v0.1: 14+ models across 7 groups; 8 comparison axes; threat ranking; architecture-novel elements with no comparator (cycle 016 S4) |
| `SPEC-20260506-004-dream3r-architecture-v02.md` | Dream3R architecture v0.2 delta spec: six numbered deltas on v0.1 (frame budget 30-50 ms/frame; C1 DINOv3-S replaces ViT-L; C2 bounded anchor bank + NSA-style retrieval A+B; sparse attention as engineering optimization; C5 Composer pool admits 7 lightweight experts; main-claim narrowed to A Verification-as-architecture + D Heterogeneous best-of-N Composer); v0.1 body unmodified per DEC-20260506-002 (cycle 018 S4) |
| `SPEC-20260506-005-dream3r-ablation-plan-v02.md` | Dream3R ablation plan v0.2 delta addendum: nine v0.2 ablations (ABL-v02-1..9) anchored to SPEC-20260506-004 v0.2 architecture deltas — NSA-removal / DINOv3 backbone tier (-S/-B/-L) / frozen vs partial-unfreeze / Composer best-of-N vs single-expert / capability_match measurement / selection-gate signal subsetting / head training schedule / frame-budget benchmark / NSA kernel decomposition. Tier 1 load-bearing: ABL-v02-1+4+6. Falsification mapping for main-claim A + D + E. Per-ABL review checklist subsection for other-agent handoff per user request "其他agent审阅修改". v0.1 body unmodified per DEC-20260506-003 (cycle 019 S2) |
| `SPEC-20260507-001-dream3r-comparator-map-v02.md` | Dream3R comparator map v0.2 delta addendum: reorganizes v0.1 16-entry comparator pool into 5 tiers per SPEC-004 Delta 5 — in-pool 7 admitted experts (MASt3R / Fast3R / Spann3R / CUT3R / MoGe-2 / DepthAnything-V2 / Test3R) + out-of-pool 3 drops (VGGT / MapAnything / Kimi-KDA per DEC-002 reasons) + out-of-scope 1 (ViT-L per Delta 2) + foundation 1 (DUSt3R) + orthogonal 8. Adds 3 NEW comparison axes 9-11 (NSA-style sparse attention / DINOv3 backbone tier / Composer expert pool composition) per Deltas 2-5. Re-ranks threats against pillar A + D narrowing — pillar A HIGH threats: Test3R alone + TTT3R; pillar D HIGH threats: VGGT offline-batch + each in-pool expert alone. Full v0.1 → v0.2 traceability matrix; 5 risks R-cm-1..5 + 5 open questions Q1-Q5 surfaced including ABL-v02-10 Test3R-alone candidate for pillar A robustness. v0.1 SPEC-003 body unmodified per DEC-20260507-001 (cycle 021 S3); v0.1 Version history v0.2 pointer entry appended. Closes v0.2 markdown trio (architecture / ablation / comparator) |
| `SPEC-20260507-002-dream3r-ablation-plan-v03-addendum.md` | Dream3R ablation plan v0.3 delta addendum on SPEC-005 (v0.2). Adds ABL-v02-10 Test3R-alone comparator (Tier 1; Q1 from comparator map; tests whether Test3R built-in verifier matches Dream3R Critic-gate pipeline on pillar A; 3 outcome scenarios; ~120 GPU-hours). Pillar A falsification coverage map (4 sub-claims × primary ABL; RA-07 from Path C Agent B). ABL-v02-4 VGGT offline-batch baseline annotation (Q2; Variant X; ~20 GPU-hours). Updated tier placement + compute budget ~1377 GPU-hours. SPEC-005 v0.2 body NOT modified (cycle 023 S3) |
| `SPEC-20260508-001-dream3r-c2-memory-v03-addendum.md` | Dream3R C2 Memory v0.3 architecture addendum. Supersedes SPEC-20260506-004 Delta 3 as current C2 memory design: vector GRU + vector AnchorBank + NSA-label becomes latent state-token recurrence + explicit spatial key/value memory + geometry-aware bus-gated writes. Reinterprets compressed/selected/sliding branches as CUT3R-like state tokens / Spann3R-like spatial bank / recent frame-value tokens. Adds state schema, bus publications/reads, prototype sequence P0-P5, memory-specific ablation candidates, risks, and paper evidence-boundary correction. Markdown only; no implementation authorized (cycle 026 S3) |
| `SPEC-20260508-002-dream3r-memory-v03-ablation-addendum.md` | Cycle 028 Memory v0.3 ablation addendum, reviewed and corrected in cycle 029. Maps P0 variants V0-V3 and claims C2M-C1..C2M-C11 to ABL-memory-0..11. Separates P0 tensor-only tests from future module/integration tests. Cycle 029 v1.1 adds oracle-bus boundary, hard/soft fail rule, narrowed C2M-C1/C8, and stronger ABL-memory-3. Engineering plan only; no implementation authorized |

### `literature/` - Paper artifacts

| File | Description |
|---|---|
| `PAPER_DRAFT_V1.md` | Dream3R paper draft v1.2 (cycle 022 update). §1-2, §4-8 = v0.1 cycle 017 baseline. §3.8 NEW: six v0.2 architecture deltas (Delta 1 frame budget / Delta 2 DINOv3-S / Delta 3 NSA anchor bank / Delta 4 attention regime / Delta 5 7-expert pool / Delta 6 A+D pillar narrowing). §6.0–6.3 NEW: v0.2 comparator positioning (5-tier pool / A+D threat table / 3 new axes). §6.4 = v0.1 threat table preserved. Source: SPEC-004 (§3.8) + SPEC-20260507-001 (§6). Per DEC-20260507-002 (cycle 022) |

### `storyboards/` - Teacher Demo Storyboards

Format: `STORY-YYYYMMDD-NNN-<slug>.md`. One file per finalist teacher demo. Created via `templates/demo_storyboard.md`. Drafting a storyboard does NOT authorize showing; showing requires a separate DEC per `AGENT_MASTER_PROMPT.md` section 6.

| File | Finalist | Status | Role |
|---|---|---|---|
| `STORY-20260505-001-critic.md` | Geometry Critic (SPEC-20260503-001) | draft | D3 first teacher demo target per DEC-20260505-001 (1); three placeholder panels on CRITIC-02 Fast3R-vs-Spann3R regime; locked surprise hook "Catch a near-failure and repair it on the spot"; showing NOT authorized |
| `STORY-20260505-002-memory.md` | Executive Memory (SPEC-20260503-002) | draft | cycle 012 (e); three placeholder panels on MEMORY-01 MonST3R 48-frame regime; locked surprise hook "Memory that knows what to drop survives a walk where memory that keeps everything drowns"; showing NOT authorized |
| `STORY-20260505-003-permanence.md` | Dynamic Object Permanence (SPEC-20260503-003) | draft | cycle 012 (e); three placeholder panels on PERMANENCE-01 + 02 + 03 portfolio; locked surprise hook "Watch the static map stay clean while the scene moves"; showing NOT authorized |
| `STORY-20260505-004-composer.md` | 3R Composer (SPEC-20260504-001) | draft | cycle 012 (e); three placeholder panels using COMPOSER-04 KYKT-metadata-derived capability_card scoreboard; locked surprise hook "Same reconstruction, less compute — when two models tie, pick the cheaper one"; showing NOT authorized |

### `registry/` - Lightweight Indexes

| File | Role |
|---|---|
| `source_registry.md` | Source ID -> title/url/track/evidence map |
| `research_unit_registry.md` | RU ID -> name/track/decision map |
| `decision_registry.md` | DEC ID -> scope/decision/status map |

### `templates/` - Reusable Forms

| File | Role |
|---|---|
| `source_card.md` | Source intake form |
| `research_unit.md` | Dream Research Unit form |
| `decision_memo.md` | Decision memo form |
| `cycle_log.md` | Cycle log form |
| `experiment_plan.md` | Experiment plan form |
| `frontend_design_handoff.md` | Frontend task brief form for Gemini CLI |
| `proxy_case_card.md` | Branch-neutral L2 proxy case-card form (P1-P8) |
| `finalist_mechanism_spec.md` | Branch-neutral mechanism spec form (requires user approval) |
| `demo_storyboard.md` | Branch-neutral teacher demo storyboard skeleton; filling does not authorize showing |

## Find By Question

| Question | Where to look |
|---|---|
| What phase are we in? | `WORKFLOW_STATUS.md` |
| What is the next user decision? | `WORKFLOW_STATUS.md` -> Recommended Next User Decision; current default is cycle 032 local `ABL-memory-1` or review of `experiments/prototypes/memory_v03_p0/outputs/` |
| What sources do we know about? | `sources/FRONTIER_SOURCE_MAP.md` and `registry/source_registry.md` |
| What ideas are on the table? | `units/RESEARCH_UNIT_BANK.md`, `units/IDEA_SCOREBOARD.md` |
| Why this branch and not that one? | `planning/BRANCH_COMPARISON_MATRIX.md`, `planning/BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| Which branches are user-approved finalists, and what are their specs? | `specs/` (one file per finalist) |
| What can Dream do without asking the user? | `AGENT_MASTER_PROMPT.md` section 6 |
| What requires user approval? | `AGENT_MASTER_PROMPT.md` section 6 + `WORKFLOW_STATUS.md` Blocked Until User Decision |
| How should I behave when synthesizing or editing files? | `paradigm/RESEARCH_CODE_DISCIPLINE.md` |
| What is the latest research result? | newest file under `cycles/`; current latest is `cycles/CYCLE-20260508-008.md` (cycle 031 local P0 scaffold + ABL-memory-0 pass) |
| What did we decide? | `registry/decision_registry.md` and files under `decisions/` |
| What experiments are planned or locally scaffolded? | files under `experiments/` plus C2 v0.3 prototype sequence in `specs/SPEC-20260508-001-dream3r-c2-memory-v03-addendum.md`, P0 plan in `planning/MEMORY_V03_P0_PROTOTYPE_PLAN.md`, reviewed ablation map in `specs/SPEC-20260508-002-dream3r-memory-v03-ablation-addendum.md`, review in `planning/MEMORY_V03_ABLATION_REVIEW.md`, execution template in `planning/MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md`, and local `ABL-memory-0` scaffold under `experiments/prototypes/memory_v03_p0/`; later ablations still require separate DEC + per-step gate |
| How should the frontend agent work? | `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md` |
| How do humans and agents cooperate? | `handoff/COLLABORATION_ROADMAP.md` |

## Convention Reminders

- Evidence labels: `paper-proven`, `code-observed`, `engineering-demonstrated`, `demo-observed`, `inferred`, `speculative`, `unknown`.
- Decision approval gates: see `AGENT_MASTER_PROMPT.md` section 6.
- ID format: `SRC-YYYY-NNN`, `MECH-YYYY-NNN`, `RU-NNN`, `DEC-YYYYMMDD-NNN`, `EXP-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`, `SPEC-YYYYMMDD-NNN`.
- Guidance file sync rule: when promoting a workflow artifact, also update `AGENT_MASTER_PROMPT.md`, `README.md`, `WORKFLOW_STATUS.md`, `RESEARCH_STATE.md`, current cycle log.
