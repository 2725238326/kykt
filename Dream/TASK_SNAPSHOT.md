# Dream Task Snapshot

Last updated: 2026-05-16 (cycle 036 closed: dual-scope packaging + proposal kickoff — Part A 3 files in `3R-mix/deliverables/` (SUBMISSION_PACKAGE_ADVISOR / SUBMISSION_RECORD / RELATION_TO_TRACK_A) package the 2026-05-15 Track B Chinese survey for advisor / school internal review; Part B 4 files in new subdirectory `planning/proposal_dream3r/` (OUTLINE_V1 9-section dual outline + STYLE_CONTRACT vocab substitution table 22 rows + sync rule + DRAFT_INTERNAL_V1 § 1 ~1800 字 + DRAFT_EXTERNAL_V1 § 1 ~1500 字 with 候选架构 X 代号) launch the Chinese Dream3R 开题报告 dual-draft scaffold; § 2-§ 9 placeholders only; WORK_RISK_REGISTER v1.1 → v1.2 additive +3 rows (R-PROP-VOCAB-1 / R-PROP-CLAIM-1 / R-PROP-SYNC-1); DEC-20260516-001 + cycle log; status flipped idle → cycle 036 in_progress → idle on close; G2 + G3a + G3b + G4 vocab firewall + over-claim greps all returned 0 hits; cycle 035 deliverables + Track A Dream3R v0.3 server-verified through cycle 034 + Track B 3R-mix wound down 2026-05-14 + 2026-05-15 prose naturalization pass all unchanged)

Status: **idle** (cycle 036 closed 2026-05-16: 13 file ops landed (9 NEW + 4 MODIFIED); Part A advisor submission packaging + Part B Dream3R proposal dual-draft kickoff + risk register v1.2 + cycle log + sync chain — no spec / code / server action; § 2-§ 9 proposal body text drafting deferred to cycle 037+; cycles 033 + 034 + 035 closed earlier; awaiting user direction on next workstream)

## Why this file exists

This file is the highest-authority entry point for any Dream session — human, Codex, or another agent. It exists so that an interrupted task can be resumed cleanly, in this conversation or in a fresh one, without context loss.

Read order on session start:

1. **This file first** (`TASK_SNAPSHOT.md`)
2. Then `AGENT_MASTER_PROMPT.md` Mandatory Load Protocol (this file is item 1 of that protocol; the rest follows)
3. Then proceed per `AGENT_MASTER_PROMPT.md`

If this file's "Status" is `in_progress` or `blocked`, do NOT start new work. Resume the named task from `if_interrupted_resume_from` first.

If this file's "Last updated" timestamp is older than the latest cycle log under `cycles/`, this file is stale; trust the cycle log and update this file before doing anything else.

## Current task

```text
task_id:    none-active
phase:      idle between cycles
cycles:     032 + 033 + 034 + 035 + 036 closed; Track B 3R-mix survey polished + packaged for advisor review; Dream3R 开题报告 dual-draft kickoff with § 1 alignment proof
status:     idle
```

One-line description:

```text
No active cycle. Two parallel tracks are at a checkpoint:
  - Track A (Dream3R v0.3 code): server-verified; first KITTI real-data
    smoke run; canonical onboarding doc REVIEW_PROMPT.md; RECENT_PROGRESS.md
    is the canonical W19-W22 ledger; NEXT_PHASE_ROADMAP.md lists post-demo
    candidates (real-data ablation table / Critic calibration / DTU loader
    / 3DGS renderer / TTT).
  - Track B (3R-mix Chinese survey, separate workspace at Dream/3R-mix/):
    18-page LaTeX manuscript with 4 paper Fig.1 crops embedded, 6
    figures (4 TikZ + 2 paper-Fig.1 composites), 5 booktabs tables, 44
    references all cited; current recommended deliverable
    `deliverables/3r_survey_stage_final_2026-05-15_natural.pdf`;
    remaining work documented in
    3R-mix/NEW_CHAT_HANDOFF.md "未完成任务".
```

## Subtask board (none active; last cycle 034 board preserved as the most recent reference)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| C034-S1 | Sync discipline via `sync_verify_server.ps1` | done | `code/dream3r/scripts/sync_verify_server.ps1` |
| C034-S2 | W15 calibration (config-threaded geometric thresholds) | done | `code/dream3r/modules.py`, `code/dream3r/config.py` |
| C034-S3 | W16 ISA pose stress tests | done | `code/dream3r/tests/test_isa_slots.py` |
| C034-S4 | W17 Mamba-Transformer hybrid recurrence | done | `code/dream3r/mamba_block.py` |
| C034-S5 | W18 GaussianHead tensor contract (no renderer) | done | `code/dream3r/gaussian_head.py` |
| C034-S6 | KITTI real-data loader + `evaluate_real_sequence` | done | `code/dream3r/data/kitti_real.py`, `evaluate_real_sequence.py` |
| C034-S7 | Synthetic ablation runner + demo export pack | done | `ablate_recurrence.py`, `export_demo_artifacts.py` |
| C034-S8 | 3R-mix Track B kickoff (LaTeX skeleton + bib + notes) | done | `Dream/3R-mix/main.tex`, `references.bib`, `notes/` |
| 0513-S1 | 3R-mix structural overhaul (10-section plan + new §9 + new `tab:testtime`) | done | `Dream/3R-mix/main.tex` |
| 0513-S2 | 3R-mix `fig:paradigm` TikZ + lineage label refresh | done | `Dream/3R-mix/main.tex` |
| 0513-S3 | 3R-mix paper Fig.1 embedding (DUSt3R / VGGT / MonST3R / CUT3R) | done | `figures/`, `main.tex` |
| 0513-S4 | 3R-mix source-checked rewrites (MV-DUSt3R+ / Fast3R / VGGT / TTT3R + 7 × 2026 preprints) | done | `Dream/3R-mix/main.tex`, `notes/paper_inventory.md` |
| 0513-S5 | 3R-mix three refinement passes (caption shortening + language naturalization + final refine) | done | `deliverables/3r_survey_stage_final_2026-05-13_refined.pdf` (16 A4 pages, 0 errors / 0 warnings) |

## Cycle 036 subtask board (closed 2026-05-16)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| C036-S1 | DEC-20260516-001 launch authorization | done | `decisions/DEC-20260516-001-cycle-036-survey-submission-and-proposal-kickoff.md` |
| C036-S2 | Part A advisor cover note (vocab-clean per G2) | done | `3R-mix/deliverables/SUBMISSION_PACKAGE_ADVISOR_2026-05-16.md` |
| C036-S3 | Part A submission record (slots for recipient / channel / SHA256 / submitted_at) | done | `3R-mix/deliverables/SUBMISSION_RECORD_2026-05-16.md` |
| C036-S4 | Part A Track A relationship internal meta (escape valve; not delivered to advisor) | done | `3R-mix/deliverables/RELATION_TO_TRACK_A_2026-05-16.md` |
| C036-S5 | Part B style contract (vocab substitution table 22 rows + bilingual sync rule + 候选架构 X naming + candidate-not-final 句式 表) | done | `planning/proposal_dream3r/STYLE_CONTRACT.md` |
| C036-S6 | Part B 9-section dual outline + chapter mapping + 字数 estimate + cycle 037+ drafting order | done | `planning/proposal_dream3r/OUTLINE_V1.md` |
| C036-S7 | Part B 内部稿 § 1 ~1800 字 + § 2-§ 9 placeholders | done | `planning/proposal_dream3r/DRAFT_INTERNAL_V1.md` |
| C036-S8 | Part B 外部稿 § 1 ~1500 字 (代号 候选架构 X) + § 2-§ 9 placeholders; G3a + G3b + G4 vocab firewall + over-claim greps all 0 hits | done | `planning/proposal_dream3r/DRAFT_EXTERNAL_V1.md` |
| C036-S9 | Cross-spec proposal-cycle risk register additions (3 new rows: R-PROP-VOCAB-1 / R-PROP-CLAIM-1 / R-PROP-SYNC-1) | done | `planning/WORK_RISK_REGISTER.md` (v1.2 additive) |
| C036-S10 | Cycle 036 log | done | `cycles/CYCLE-20260516-001.md` |
| C036-S11 | Sync chain (TASK_SNAPSHOT first + WORKFLOW_STATUS + INDEX) | done | this file + `WORKFLOW_STATUS.md` + `INDEX.md` |

Cycle 036 deliverable summary:

```text
DEC-20260516-001:
  - authorizes only 13 file ops total (9 NEW + 4 MODIFIED): 3 Part A
    files in 3R-mix/deliverables/ + 4 Part B files in new
    planning/proposal_dream3r/ + DEC + cycle log + 4 sync targets
    (WORK_RISK_REGISTER v1.1 -> v1.2 additive + TASK_SNAPSHOT +
    WORKFLOW_STATUS + INDEX)
  - forbids Dream/3R-mix/main.tex / references.bib / notes/* edits,
    Dream/specs/ edits, Dream/code/ edits, Dream/paradigm/ edits,
    server actions, checkpoint, training, model inference, real
    submission action (left to user post-cycle), v0.4 spec delta
    drafting (B1/B2/B3 from cycle 035 proposal §5 remain
    proposal-status; each requires its own DEC), § 2-§ 9 proposal
    body text drafting (§ 1 only this cycle as alignment proof),
    Dream-vocabulary in advisor cover note, raw "Dream3R" /
    forbidden patterns in DRAFT_EXTERNAL_V1.md § 1 prose

Part A (3R-mix advisor submission packaging):
  - SUBMISSION_PACKAGE_ADVISOR_2026-05-16.md ~600 字 Chinese cover
    note; 6 sections (主旨 / 范围 / 与英文综述差异 / 证据边界 /
    路线说明 / 请求审阅事项); G2 vocab firewall grep 0 hits on
    Dream|Dream3R|KYKT|agent|skill|workflow|本地项目|cycle|SPEC-|DEC-|CR-
  - SUBMISSION_RECORD_2026-05-16.md YAML metadata + checklist with
    slots for recipient / channel / pdf_sha256 (PowerShell
    Get-FileHash) / submitted_by / contact / submitted_at;
    pre-filled fields page_count = 18, ref_count = 44, figure_count
    = 6, table_count = 5, vocab_grep_verified = 2026-05-16
  - RELATION_TO_TRACK_A_2026-05-16.md ~600 字 internal meta;
    documents Track B / Track A relationship; not delivered with
    PDF; only file allowed to mention Dream-vocabulary

Part B (Dream3R proposal dual-draft kickoff):
  - new subdirectory planning/proposal_dream3r/ created
  - STYLE_CONTRACT.md vocab substitution table 22 rows
    (Dream3R -> 候选架构 X / SPEC-* -> 体系结构设计文档 v0.X /
    DEC-* -> 项目关键决策点 N / CYCLE-* -> 研发周期 N /
    CR-1..CR-6 -> 信号校验规则族 (1-6) / W1..W22 -> 实现里程碑 1-22 /
    F-001/F-002 -> 内部工作规则 / 算力部署约束 / agent / skill /
    workflow / KYKT / ablate_recurrence.py / ABL-memory-N /
    ABL-v02-N / NSA three-branch / AnchorBank / StateToken / Mamba
    hybrid / pointmap L2 = 20.47 / 4DGS asset etc.); §3 sync rule
    internal-is-master + periodic external snapshot + grep
    verification + cycle-end sync log; §4 候选架构 X naming
    introduction; §5 candidate-not-final 句式 contrast (9 禁用 vs
    允许 句式 对照); §6 sync log (cycle 036 entry: 13 vocab
    substitutions seeded + § 1 grep verified clean)
  - OUTLINE_V1.md §2 9-section dual outline (外稿 ~16000 字 / 内稿
    ~21100 字) + §3 chapter mapping table (外稿 ↔ 内稿 ↔ 复用
    素材) + §4 cycle 037-042 drafting order (cycle 037 §2 国内外
    研究现状 first because largest single-section block + most
    heavily reuses Track B 综述 + double-draft sync stress-tests
    STYLE_CONTRACT immediately) + §5 §1 风格样本 200 字 双稿对照
  - DRAFT_INTERNAL_V1.md §1 ~1800 字 covers §1.1 Track A 主线决策
    起源 (DEC-20260506-001) / §1.2 Dream3R v0.3 当前状态 (W1-W18 +
    KITTI smoke L2 = 20.47 + 部署服务器 path) / §1.3 Track B 综述
    四轴反哺 / §1.4 三个核心研究问题 Q1 验证机制路径 (Critic) +
    Q2 长序列内存路径 (Memory) + Q3 多专家组合路径 (Composer) /
    §1.5 候选 vs 最终边界 / §1.6 Dream 项目工件引用 (DEC + SPEC +
    cycle 链); §2-§9 placeholders with TBD comments + 子节
    suggestions; G4 over-claim grep 0 hits after §1.5 rephrase
    "本研究的成果不是论证 Dream3R 相对 SOTA 具有压倒性优势, 而是
    评估..."
  - DRAFT_EXTERNAL_V1.md §1 ~1500 字 covers §1.1 前馈式三维重建
    (3R) 研究方向 / §1.2 六类典型几何失败模式 (弱纹理 / 镜面玻璃
    / 快速运动 / 长基线 / 尺度漂移 / 域外) / §1.3 三组未充分解决
    问题 (验证 vs 适应 + 长序列内存机制统一 + 多专家组合实证) /
    §1.4 本研究目标 (代号 候选架构 X 引入 + 4 设计目标) / §1.5
    研究地位 (candidate-not-final + 不押注单一方案) / §1.6 学术
    价值与意义 (3 方面贡献); §2-§9 placeholders; G3a vocab firewall
    grep 0 hits after fixing §元数据 row "完全剥离内部 workflow
    词汇" -> "完全剥离内部研究流程相关用词"; G3b "Dream3R"
    case-insensitive grep 0 hits after removing §元数据 文件路径
    row containing workspace path; G4 over-claim grep 0 hits after
    §1.5 rephrase "本研究的目标不是论证 X 相对现有方法具有压倒性
    优势..."

WORK_RISK_REGISTER.md v1.2 additive (+3 rows):
  - R-PROP-VOCAB-1: external draft Dream-vocabulary leakage; mitigated
    by STYLE_CONTRACT §2 vocab table + §3 sync rule + per-sync grep
    verification; cycle 036 close passed verification with 0 hits on
    full forbidden pattern
  - R-PROP-CLAIM-1: 开题报告 over-claim 候选架构 X 为最终方案;
    mitigated by STYLE_CONTRACT §5 candidate-not-final 句式 表 +
    per-cycle grep verification on draft sections as they land;
    cycle 036 close passed verification with 0 hits on both internal
    and external §1
  - R-PROP-SYNC-1: 双稿语义漂移 (内部稿 §X vs 外部稿 §X 对同一
    研究问题描述出现实质差异); mitigated by STYLE_CONTRACT §3
    internal-is-master sync rule + 外部稿 standalone 编辑限制 +
    每 cycle 末尾 sync log entry

Result:
  - 13 file ops complete (9 NEW + 4 MODIFIED); stop gates G0-G6 all
    passed; G2 + G3a + G3b + G4 vocab firewall + over-claim greps
    all returned 0 hits after one corrective pass each; mainline
    decisions all in force

Evidence boundary:
  - packaging + planning + § 1 markdown only; no actual survey
    submission performed (manual user action post-cycle), no § 2-§ 9
    proposal body text drafted, no spec change, code change,
    calibration run, or ablation run validated by cycle 036
  - Track B 3R-mix manuscript surface unchanged (still wound down at
    2026-05-15 prose naturalization deliverable; only `deliverables/`
    received 3 new files)
  - candidate-not-final boundary preserved: DRAFT_INTERNAL_V1 §1.5 +
    DRAFT_EXTERNAL_V1 §1.5 explicitly state X / Dream3R is being
    evaluated, not converged on

Next admissible direction (per DEC-20260516-001 §Next Direction If Passed):
  A. launch cycle 037 § 2 国内外研究现状 (recommended; largest single
     block; double-draft sync stress-test)
  B. user executes actual survey submission action + fills
     SUBMISSION_RECORD slots (manual action outside any cycle)
  C. revise OUTLINE_V1 chapter structure before cycle 037 (preserve
     V1, create V2)
  D. pause + revise cycle 036 deliverables based on quality review
  E. return to architecture-first mainline non-proposal work
     (W22 / W23 / Fast3R omegaconf per cycle 035 §Next Direction D)
  F. launch one of cycle 035 §Next Direction A-C instead (calibration
     / long-seq ablation / v0.4 spec delta) -> independent DEC each
```

## Cycle 035 subtask board (closed 2026-05-15)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| C035-S1 | Survey-driven optimization proposal (cycle 035 upstream) | done | `planning/SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL.md` (status preserved at draft) |
| C035-S2 | DEC-20260515-001 launch authorization | done | `decisions/DEC-20260515-001-cycle-035-survey-driven-markdown-deliverables-launch.md` |
| C035-S3 | SOTA matrix V2 (re-label SPEC-007 v0.2 Tier 1-5 against survey four-axis + input-extension bonus axis) | done | `planning/SOTA_MATRIX_V2.md` |
| C035-S4 | Critic calibration plan V1 (per-failure-mode threshold standardization, plan-only) | done | `planning/CRITIC_CALIBRATION_PLAN_V1.md` |
| C035-S5 | Long-seq real-data table plan (ablate_recurrence extension to KITTI ≥10 windows, plan-only) | done | `planning/LONG_SEQ_REAL_TABLE_PLAN.md` |
| C035-S6 | Cross-spec risk register additions (4 new rows: R-OOD-1 / R-EXT-PRIOR-1 / R-4DGS-LIC-1 / R-INPUT-EXT-1) | done | `planning/WORK_RISK_REGISTER.md` (v1.1 additive) |
| C035-S7 | Cycle 035 log | done | `cycles/CYCLE-20260515-001.md` |
| C035-S8 | Sync chain (TASK_SNAPSHOT first + WORKFLOW_STATUS + INDEX) | done | this file + `WORKFLOW_STATUS.md` + `INDEX.md` |

Cycle 035 deliverable summary:

```text
DEC-20260515-001:
  - authorizes only writing 3 new planning files + appending 4 risk rows
    to WORK_RISK_REGISTER + sync chain + cycle log
  - forbids Dream/specs/ edits, Dream/code/ edits, server actions,
    checkpoint, training, frontend, ablation runs, evaluate_real_sequence
    runs, Track B 3R-mix edits, RECENT_PROGRESS / NEXT_PHASE_ROADMAP edits,
    v0.4 spec delta drafting (B1/B2/B3 from proposal remain proposal-status)

3 new planning files (cycle 035 P0-1/P0-2/P0-3 deliverables):
  - SOTA_MATRIX_V2.md: re-labels 19 comparator entries (T1 in-pool 7
    + T2 dropped 3 + T3 oos 1 + T4 foundation 1 + T5 orthogonal 8) plus
    Point3R / Mem3R / G-CUT3R / Pow3R / MASt3R-SfM appendix entries against
    five axes (failure modes / long-seq memory / test-time / output asset
    + input extension bonus); identifies 4 first-class-support gaps
  - CRITIC_CALIBRATION_PLAN_V1.md: maps survey six failure modes to C4
    Critic five sub-signals; defines sub-sample sampling rules per mode;
    outlines method A (distribution-quantile P95) vs method B (supervised
    classifier) with selection decision tree; sets 5-metric validation gate
  - LONG_SEQ_REAL_TABLE_PLAN.md: maps 4 ablate_recurrence variants
    (baseline_cross_attention / mamba_hybrid / no_nsa / no_stable_memory)
    to survey §6 four memory mechanism types; defines 4 long-seq-specific
    metrics (scale_drift_proxy / memory_decay_proxy / anchor_fill_rate /
    retrieval_diversity); outlines windows=10/20/50/100 staged execution
    with 6-metric validation gate; explicit B4 budget-governance subtype
    gap noted

WORK_RISK_REGISTER.md v1.1 additive:
  - R-OOD-1: OOD detection path absent in C4 Critic
  - R-EXT-PRIOR-1: external prior vs geometry conflict unmodeled in
    CR-1..CR-6
  - R-4DGS-LIC-1: 4DGS asset license chain undocumented in W18 GaussianHead
  - R-INPUT-EXT-1: input extension axis (pose / sparse depth / video) absent

Result:
  - 7 artifacts present (DEC + cycle log + 3 plans + risk register + this
    sync chain); stop gates G0-G5 all passed; visual scan confirmed no
    forbidden-action claims in the 3 new files

Evidence boundary:
  - planning markdown only; no calibration run, no ablation run, no spec
    change, no code change, no server action validated by cycle 035
  - 3 new plans explicitly mark themselves "plan-only; execution needs
    independent DEC" in their §1 + Metadata sections
  - proposal upstream status remains draft until separate DEC formally
    accepts it as v0.4 design input

Next admissible direction (per DEC-20260515-001 §Next Direction If Passed):
  A. calibration data collection on KITTI -> requires independent DEC + F-002
  B. ablate_recurrence on KITTI long windows -> requires independent DEC + F-002
  C. v0.4 spec delta drafting (B1 Critic path split / B2 output asset
     contract / B3 input extension axis) -> requires independent DEC per delta
  D. pause + revise proposal based on cycle 035 deliverable quality
  E. return to architecture-first mainline non-survey work (W22 / W23)
```

Cycle 031 active boundary:

```text
Authorized user trigger:
  - "那就先进行后续操作吧"

Execution scope:
  - local deterministic P0 fixture scaffold
  - ABL-memory-0 only as validity gate
  - output manifest/log/metric/summary/evidence-boundary artifacts

Explicit exclusions:
  - no ABL-memory-1..8 performance claims in this cycle
  - no ABL-memory-9..11
  - no server/runtime/model/checkpoint/training/frontend work
```

Cycle 031 deliverable summary:

```text
DEC-20260508-007:
  - authorizes only local P0 scaffold + ABL-memory-0 validity gate
  - forbids Dream/code, server integration, model imports, checkpoint use,
    training, frontend work, paper claim promotion, and ABL-memory-1..11
    behavior claims

experiments/prototypes/memory_v03_p0/:
  - deterministic fixture generator
  - oracle-bus contract
  - raw-label exclusion audit
  - ABL-memory-0 gate runner
  - direct smoke test

outputs:
  - fixtures_manifest.json
  - write_log.jsonl
  - metrics_abl_memory_0_8.csv
  - summary_go_no_go.md
  - evidence_boundary_update.md

Result:
  - ABL-memory-0 pass, 22/22 validity checks
  - pytest unavailable in the current Python environment
  - direct smoke test passed with python tests\test_abl_memory_0.py

Evidence boundary:
  - fixture/logging substrate only
  - no memory quality, retrieval quality, recurrence quality, reconstruction
    quality, server behavior, model behavior, or paper claim validated
```

Cycle 030 deliverable summary (for resume context):

```text
DEC-20260508-006:
  - accepts only markdown-only template creation scope
  - authorizes planning/MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md
  - keeps actual P0 implementation gated

planning/MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md:
  - predefines future DEC fields, allowed paths, forbidden actions,
    ABL-memory-0..8 scope, ABL-memory-9..11 exclusion, oracle-bus
    rules, required outputs, stop gates, result labels, go/no-go rules,
    and post-execution evidence boundary

Evidence boundary:
  - cycle 030 is authorization-template evidence only
  - no ABL-memory result, prototype implementation, C2 Memory v0.3
    quality, reconstruction quality, spatial retrieval quality,
    state-token recurrence performance, or paper claim is validated yet

Next expected research object after cycle 030:
  - user decision on whether to authorize local static tensor P0
    execution for ABL-memory-0..8, revise the template, or return to
    research design
```

## Last completed task pass

```text
pass_name:        Cycle 031 close pass (Memory v0.3 local P0 scaffold
                  + ABL-memory-0 gate done in single session 2026-05-08)
date:             2026-05-08
trigger:          User authorized proceeding after cycle 030 template and
                  GitHub CLI discussion.
files_modified:   TASK_SNAPSHOT.md, WORKFLOW_STATUS.md, RESEARCH_STATE.md,
                  INDEX.md, README.md, AGENT_MASTER_PROMPT.md,
                  registry/decision_registry.md
new_artifacts:    decisions/DEC-20260508-007-cycle-031-p0-local-static-
                  tensor-scaffold.md
                  experiments/prototypes/memory_v03_p0/
                  cycles/CYCLE-20260508-008.md
result:           Local P0 scaffold created. ABL-memory-0 passed 22/22
                  fixture/logging validity checks and wrote required
                  outputs under experiments/prototypes/memory_v03_p0/outputs/.
paper_boundary:   No paper evidence promoted. Cycle 031 validates only the
                  local fixture/logging substrate.
discipline:       No Dream/code edit. No server integration. No model run.
                  No checkpoint use. No training. No frontend.
verification:     python run_ablations.py --output outputs returned pass;
                  python -m pytest tests failed because pytest is not
                  installed; python tests\test_abl_memory_0.py passed.

pass_name:        Cycle 030 close pass (Memory v0.3 P0 execution DEC
                  template done in single session 2026-05-08)
date:             2026-05-08
trigger:          User asked to continue according to the current plan.
files_modified:   TASK_SNAPSHOT.md, WORKFLOW_STATUS.md, RESEARCH_STATE.md,
                  INDEX.md, README.md, AGENT_MASTER_PROMPT.md,
                  registry/decision_registry.md
new_artifacts:    decisions/DEC-20260508-006-cycle-030-p0-execution-
                  dec-template.md
                  planning/MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md
                  cycles/CYCLE-20260508-007.md
result:           P0 execution DEC template completed. It predefines
                  future DEC fields, allowed local prototype path,
                  forbidden server/model/checkpoint paths, allowed and
                  forbidden actions, ABL-memory-0..8 scope,
                  ABL-memory-9..11 exclusion, oracle-bus boundary,
                  required outputs, stop gates, result labels, go/no-go
                  rules, and post-execution evidence boundary.
paper_boundary:   No paper evidence promoted. Cycle 030 is
                  authorization-template evidence only.
discipline:       Markdown-only. No server code edit. No model run.
                  No training. No checkpoint download. No frontend.
verification:     git diff --check returned no whitespace errors
                  (line-ending warnings only); stale-pointer search
                  returned no active hits; checked markdown fence
                  counts for edited key files and all were even.

prior_pass_name:  Cycle 029 close pass (Memory v0.3 ablation review
                  and correction done in single session 2026-05-08)
prior_pass_date:  2026-05-08
prior_pass_files: TASK_SNAPSHOT.md, WORKFLOW_STATUS.md, RESEARCH_STATE.md,
                  INDEX.md, README.md, AGENT_MASTER_PROMPT.md,
                  registry/decision_registry.md,
                  specs/SPEC-20260508-002-dream3r-memory-v03-
                  ablation-addendum.md,
                  decisions/DEC-20260508-005-cycle-029-memory-
                  ablation-review.md,
                  planning/MEMORY_V03_ABLATION_REVIEW.md,
                  cycles/CYCLE-20260508-006.md

prior_pass_name:  Cycle 028 close pass (Memory v0.3 ablation addendum
                  done in single session 2026-05-08)
prior_pass_date:  2026-05-08
prior_pass_files: TASK_SNAPSHOT.md, WORKFLOW_STATUS.md, RESEARCH_STATE.md,
                  INDEX.md, README.md, AGENT_MASTER_PROMPT.md,
                  registry/decision_registry.md,
                  planning/MEMORY_V03_P0_PROTOTYPE_PLAN.md,
                  specs/SPEC-20260508-001-dream3r-c2-memory-v03-
                  addendum.md, decisions/DEC-20260508-004-cycle-
                  028-memory-ablation-addendum.md,
                  specs/SPEC-20260508-002-dream3r-memory-v03-
                  ablation-addendum.md, cycles/CYCLE-20260508-005.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
CURRENT RESUME OVERRIDE (cycle 032 closed; 2026-05-09):

1. Read this file (you are here).

2. Read code/dream3r/REVIEW_PROMPT.md — this is the canonical
   onboarding document for the v0.3 codebase. It contains the
   file map, architecture diagram, key contracts, known gaps,
   verification commands, and review checklist.

3. The v0.3 codebase is server-verified at
   /hdd3/kykt26/code/dream3r/dream3r/. All smoke tests (9/9),
   unit tests (4/4), profiling (8.4ms p95), and synthetic
   training (10 epochs, loss converging) pass.

4. Default next actions (user decision required):
   A. Start ablation experiments (ABL-memory-1..8) using the
      validated training pipeline on synthetic data.
   B. Connect expert adapters to real KYKT runners on server
      (MASt3R, Fast3R, etc. already have conda envs).
   C. Implement DTU dataset loader for real-data training.
   D. Add standard depth evaluation metrics (AbsRel, RMSE, etc.).
   E. Pause and return to research design / paper writing.

5. Known architecture gaps (see REVIEW_PROMPT.md "Known gaps"):
   A4 (points3d in AnchorBank), A5 (DINOv3 backbone),
   A6 (Test3R lazy invocation), C1 (DTU stub), D1-D4 (metrics),
   E1 (streaming orchestration), E2 (expert adapter stubs).

6. Hard rules from prior cycles still apply:
   - No reproduction / checkpoint download / training on real data
     without explicit user approval.
   - DEC-20260501-004 (candidate-not-final) and
     DEC-20260504-002 (no-all-in) still in force.
   - F-002: server-side execution only; local = editing + markdown.
```
3. Read experiments/prototypes/memory_v03_p0/README.md and
   experiments/prototypes/memory_v03_p0/outputs/summary_go_no_go.md.

4. Read cycles/CYCLE-20260508-008.md for the cycle 031 result and
   verification boundary.

5. Read planning/MEMORY_V03_P0_EXECUTION_DEC_TEMPLATE.md and
   decisions/DEC-20260508-006-cycle-030-p0-execution-dec-template.md
   for the parent authorization template.

6. Read decisions/DEC-20260508-005-cycle-029-memory-
   ablation-review.md and planning/MEMORY_V03_ABLATION_REVIEW.md.

7. Read planning/MEMORY_V03_P0_PROTOTYPE_PLAN.md and
   specs/SPEC-20260508-002-dream3r-memory-v03-ablation-
   addendum.md. The latter has cycle 029 v1.1 corrections.

8. Read specs/SPEC-20260508-001-dream3r-c2-memory-v03-
   addendum.md for the current C2 Memory architecture direction.

9. Default next action is a user decision on:
   A. start cycle 032 local ABL-memory-1 vector AnchorBank baseline,
   B. review ABL-memory-0 outputs before later ablations, or
   C. pause execution and return to research design.

10. ABL-memory-1..8 execution, server code edit, model run, checkpoint
    use, training, or paper claim promotion requires a separate DEC and
    per-step gate.

11. The older cycle 022 / cycle 015 resume material below is retained
    as historical traceability only. It is not the active next action.

1. Read this file (you are here).

2. Read decisions/DEC-20260507-002-cycle-022-path-c-reattempt-
   and-paper-s3s6-rewrite.md — this is the most recent strategic
   decision and documents the cycle 022 combined scope (Path C
   reattempt + paper §3+§6 v0.2 rewrite; both DONE; see below).

3. Read cycles/CYCLE-20260507-002.md — this is the cycle 022 log
   documenting: Path C SUCCEEDED (API gateway recovered; both
   agents returned 5-section reviews); 7 review-action items
   RA-01..07 captured for v0.3 addenda (conditional cycle 023.5);
   paper PAPER_DRAFT_V1.md updated to v1.2 (§3.8 + §6.0–6.3
   added; §3.1–3.7 + §6.4 preserved). Cycle 022 status = DONE.

4. Read decisions/DEC-20260506-004 (cycle 020 combined planning),
   decisions/DEC-20260506-003 (cycle 019 ablation plan v0.2),
   decisions/DEC-20260506-002 (cycle 018 v0.2 architecture deltas),
   decisions/DEC-20260506-001 (mainline architecture-first) for
   parent-cycle context.

5. Read these cycle 021 deliverables (already done in this pass):
   - specs/SPEC-20260507-001-dream3r-comparator-map-v02.md (S3;
     ~880 lines; closes v0.2 markdown trio; 5-tier reorganization
     + 3 NEW axes + threat re-rank; 5 risks + 5 open questions
     including ABL-v02-10 Test3R-alone candidate)
   - cycles/CYCLE-20260507-001.md (S4; cycle 021 log; status
     done-with-S2-blocked-by-infrastructure)

6. Reference SPEC bodies ONLY when needed (do NOT re-Read full
   files; cite by section + line anchor):
   - specs/SPEC-20260506-004-dream3r-architecture-v02.md (v0.2
     architecture; six deltas; 882 lines)
   - specs/SPEC-20260506-005-dream3r-ablation-plan-v02.md (v0.2
     ablation plan; nine ABL-v02; 991 lines)
   - specs/SPEC-20260506-003-dream3r-comparator-map.md (v0.1
     comparator map substrate; 625 lines; preserved unchanged
     in body; only Version history tail received v0.2 pointer)
   - specs/SPEC-20260506-001-dream3r-architecture.md (v0.1
     architecture; 1821 lines; do NOT re-Read)
   - planning/DREAM3R_V02_CODE_STRUCTURE.md (1086 lines; cycle
     020 deliverable; Path C review TARGET — deferred)
   - planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md (1226 lines;
     cycle 020 deliverable; Path C review TARGET — deferred)
   - code/dream3r/PLAN.md (v0.1 user-authored implementation
     roadmap; preserved unchanged)

7. Read C:\Users\27252\.claude\projects\e--kykt\memory\
   feedback_dream_mainline_architecture_first.md and
   feedback_kykt_server_topology.md (cross-session memories;
   F-002 anchor — KYKT 3R model code runs server-side at
   /hdd3/kykt26/; local Windows is markdown + orchestration
   only).

```text
Project state at this snapshot:
   Cycle 015 PAUSED at S9 done (NOT closed; NOT abandoned;
                                infrastructure reusable as future
                                Critic A4 evidence anchor).
   Cycle 016 DONE.
   Cycle 017 DONE (paper draft v1; needs v0.2 update later).
   Cycle 018 DONE (v0.2 architecture deltas; SPEC-20260506-004
                   written; six deltas; main-claim narrowed to
                   A+D).
   Cycle 019 DONE (v0.2 ablation plan addendum; SPEC-20260506-
                   005 written; 9 ABL-v02; per-ABL review
                   checklist for other-agent handoff).
   Cycle 020 DONE (combined v0.2 code structure + implementation
                   roadmap planning artifacts; DREAM3R_V02_CODE_
                   STRUCTURE.md + DREAM3R_V02_IMPLEMENTATION_
                   ROADMAP.md NEW; trajectory revision per user
                   "高强度推进").
   Cycle 021 DONE-WITH-S2-BLOCKED:
                S1 done (DEC-20260507-001 written ~430 lines;
                   cycle 021 launch + combined scope + Path C
                   protocol + 5 open questions Q1-Q5)
                S2 BLOCKED (Path C activation; 4 sub-agent
                   attempts × ~3 min each all failed with API
                   500 nil-pointer panic on Calcium-Ion/new-api
                   gateway; deferred to cycle 022 per Honesty
                   Override Option β; full incident in
                   CYCLE-20260507-001 §"Path C activation
                   incident")
                S3 done (specs/SPEC-20260507-001-dream3r-
                   comparator-map-v02.md written ~880 lines;
                   closes v0.2 markdown trio: architecture +
                   ablation + comparator)
                S4 done (cycles/CYCLE-20260507-001.md NEW)
                S5 done (full sync chain)
   Cycle 022 DONE:
                S1 done (DEC-20260507-002 written; combined
                   cycle scope lock + protocols + post-022
                   trajectory)
                S2 done (Path C SUCCEEDED — both agents 38.2s
                   + 47.5s; API gateway recovered; 7 RA items
                   captured: RA-01..04 CODE_STRUCTURE gaps +
                   RA-05..07 IMPLEMENTATION_ROADMAP risks)
                S3 done (literature/PAPER_DRAFT_V1.md v1.2;
                   §3.8 six v0.2 deltas NEW + §6.0–6.3 v0.2
                   comparator positioning NEW; §3.1–3.7 +
                   §6.4 preserved)
                S4 done (cycles/CYCLE-20260507-002.md NEW)
                S5 done (full sync chain)

v0.2 markdown trio + paper status (post cycle 022):
   - Architecture: SPEC-20260506-004 v0.2 (cycle 018; 6 deltas;
     A+D pillars)
   - Ablation plan: SPEC-20260506-005 v0.2 (cycle 019; 9 ABL-
     v02 with per-ABL review checklist)
   - Comparator map: SPEC-20260507-001 v0.2 (cycle 021 S3;
     5-tier reorganization + 3 new axes + threat re-rank)
   - Paper: PAPER_DRAFT_V1.md v1.2 (cycle 022 S3; §3.8 + §6.0–
     6.3 added; §3.1–3.7 + §6.4 v0.1 preserved)
   All 4 artifacts are now v0.2-coherent. v0.1 bodies of all 3
   substrate specs preserved unchanged per Discipline rule 5.

Historical post-022 projection (superseded by actual cycles 023-027):
   Cycle 023 actual: v0.3 planning addenda + ablation plan v0.3
              addendum (DEC-20260507-003).
   Cycle 024 actual: server-side v0.2 scaffold / engineering smoke
              baseline (DEC-20260508-001), later bounded by cycle 026.
   Cycle 025 actual: C2 memory mechanism study
              (planning/MEMORY_V03_DESIGN_STUDY.md).
   Cycle 026 actual: C2 Memory v0.3 addendum and guidance correction
              (SPEC-20260508-001 + DEC-20260508-002).
   Cycle 027 actual: P0 static tensor prototype plan
              (planning/MEMORY_V03_P0_PROTOTYPE_PLAN.md +
              DEC-20260508-003).

Resume action when user returns:
   Cycle 027 is DONE. No active task. Status `idle`. Prefer a new
   markdown-only memory-specific ablation addendum unless the user
   explicitly authorizes a separate P0 execution DEC.

   If user asks about Path C findings:
   - Agent A (CODE_STRUCTURE): 3 HIGH/MEDIUM gaps (latency budget
     absent; ExpertAdapter ABC unresolved; NSA output combination
     underspecified); 1 LOW gap (losses.py labeling contradiction).
     Full details in CYCLE-20260507-002 §S2 / RA-01..04.
   - Agent B (IMPLEMENTATION_ROADMAP): Pillar A faithfulness gap;
     3 risks (checkpoint inventory; NSA kernel cu121; T-v02-F
     oversized). Full details in CYCLE-20260507-002 §S2 / RA-05..07.
   - All 7 RA items captured for conditional cycle 023.5 v0.3
     addenda. NOT actioned in cycle 022 (B-roadmap-F rule).

   Any execution step launches only with explicit user direction.
   Do NOT propose training, checkpoint download, GPU runs, KYKT
   navigation change, frontend implementation, demo storyboard
   promotion past `draft`, thesis finalization, or retiring of any
   non-finalist track. DEC-20260501-004 candidate-not-final +
   DEC-20260504-002 no-all-in still in force.

Hard rules carried (unchanged from prior cycles):
   - No training. No checkpoint download. No reproduction. No
     KYKT navigation change. No frontend implementation. No
     thesis finalization. No retiring of any non-finalist track.
     No demo storyboard promotion past `draft`. No teacher-
     demo readiness claim.
   - DEC-20260501-004 (Dream3R candidate-not-final) and
     DEC-20260504-002 (no-all-in) still in force.
   - Cycles 021-022 were markdown only. Any code touch (whether
     per T-v02-N task or otherwise) requires a separate DEC +
     per-step micro gates per F-002 + reviewer authorization
     per IMPLEMENTATION_ROADMAP B-roadmap-F.
   - Honesty Override: VGGT offline-batch threat to pillar D
     acknowledged in PAPER_DRAFT_V1.md §6.2 and in
     CYCLE-20260507-002; no ablation numbers manufactured; all
     v0.2 paper claims carry evidence labels; Path C Agent
     findings documented verbatim without cherry-picking.
   - Trajectory adherence: paper v1.2 delivered as planned per
     cycle 022 scope; Path C SUCCEEDED; RA items captured;
     v0.3 addendum deferred to conditional cycle 023.5.
   - Per-task review checklist pattern (cycle 020) first real
     exercise SUCCEEDED in cycle 022. Path C = operational and
     confirmed working. API gateway stable at time of cycle 022.

Honor F-001 working rules throughout: do not Read large files
already cited in this snapshot; prefer Grep -n + Edit over full-
file Read + Write; cap large files in active context at <=2
simultaneously. Honor F-002: KYKT 3R model work runs server-side;
default to ssh + reuse before installing; check ssh_runner.py:22-44
ServerConfig before asking for SSH details. Cycles 022 is markdown
only and stayed local.
```

Project state at this snapshot:
   Cycle 015 PAUSED at S9 done (NOT closed; NOT abandoned;
                                infrastructure is reusable as
                                future Critic A4 evidence anchor).
   Cycle 016 IN PROGRESS:
                                S1 done (DEC-20260506-001 + memory +
                                  prior snapshot redirect block)
                                S2 done (architecture spec v0.1
                                  written 2026-05-06; 1821 lines;
                                  95 KB; specs/SPEC-20260506-001-
                                  dream3r-architecture.md)
                                S3 done (ablation plan v0.1
                                  written 2026-05-06; 10 ablations
                                  in 3 tiers; specs/SPEC-20260506-
                                  002-dream3r-ablation-plan.md)
                                S4 done (comparator map v0.1
                                  written 2026-05-06; 14+ models;
                                  specs/SPEC-20260506-003-dream3r-
                                  comparator-map.md)
                                S5 in progress (TASK_SNAPSHOT
                                  updated; cycle log + remaining
                                  sync files pending)

Mainline redirect summary:
   - Old implicit framing: framework-first paper output.
   - New explicit framing: architecture-first; Dream3R architecture
     spec is the PRIMARY output; paper is SUPPORT.
   - Cycle 015 L3 measurement work is SUPPORT / prereq for the
     architecture spec, NOT mainline.
   - Train-first remains deferred / blocked.
   - DEC-501-004 (Dream3R candidate-not-final) and DEC-504-002
     (no-all-in) still in force.

Resume action when user returns:
   Primary path: cycle 016 is nearly complete. S5 sync chain
     is in progress (TASK_SNAPSHOT updated; remaining: cycle log,
     decision registry, other sync files). Once S5 finishes,
     cycle 016 closes.

   Candidate next actions after cycle 016 closes:
     - Review S2 architecture spec + S3 ablation plan + S4
       comparator map (user can read all three and give feedback)
     - v0.2 architecture spec revision (if review or ablation
       plan surfaces substrate/bus/module issues)
     - Paper rewrite to feature Dream3R architecture as central
       claim (per architecture spec Q6)
     - Resume cycle 015 G_run for measured Critic A4 evidence
       (per architecture spec Q5)
     - A7 Cross-Modal / A8 Active Perception spec drafting
       (per architecture spec Q3)
     - Training authorization (requires separate DEC)

   S2 spec Q1..Q6 resolution status:
     Q1 substrate ablation priority -> RESOLVED in S3 ablation
        plan: hybrid first; SSM-only second; transformer-only
        third. ABL-2 in SPEC-20260506-002 specifies all three.
     Q2 adversarial vs natural CR-rule triggering -> RESOLVED
        in S3: both (B1-B5 for performance, B6 for CR-rule
        firing verification).
     Q3..Q6 -> still open; surface during user review of cycle
        016 deliverables.
   Do NOT execute anything beyond markdown drafting without
   explicit user `Go`. Do NOT propose training. Do NOT propose
   thesis finalization. Do NOT promote any of the 4 storyboards
   past `draft`.

Hard rules still in force (verbatim from S2 spec Boundaries):
   - All 4 finalist demo storyboards (STORY-20260505-001..004)
     remain markdown `draft`. Do NOT promote any to `approved-for-
     showing` without a separate DEC.
   - Do NOT start any non-Critic L3 pilot (Memory / Permanence /
     Composer L3) / training / KYKT navigation change / frontend
     implementation without explicit user approval per
     AGENT_MASTER_PROMPT.md section 6.
   - Cycle 015 has narrow exceptions ONLY for the Critic L3 pilot
     scope; everything else stays gated.
   - DEC-20260506-001 authorizes architecture-spec DESIGN +
     ablation PLANNING; NOT training, NOT GPU runs, NOT checkpoint
     creation.
   - v0.1 spec carries 13 explicit boundaries in its "Boundaries"
     section; all in force.

Honor F-001 working rules throughout: do not Read large files
already cited in this snapshot; prefer Grep -n + Edit over full-
file Read + Write; cap large files in active context at <=2
simultaneously. Honor F-002: KYKT 3R model work runs server-side;
default to ssh + reuse before installing; check ssh_runner.py:22-44
ServerConfig before asking for SSH details.
```

If this snapshot's Status is `idle`, both cycle 015 (Critic L3 measurement) and cycle 016 (Dream3R architecture spec drafting) are closed; the next live phase requires a separate user direction.

## Open user decisions (resolution status, 2026-05-04)

D1'-D4' were delegated to the agent by user message "D1-D4 你自己决策吧，有问题我们商讨" and locked in `decisions/DEC-20260504-003-cycle-009-launch.md`. Summary:

```text
1. Cycle 009 ordering (D1')        -> parallel (Composer + Critic; cross-
                                       spec contract v1 is the test path).
2. Composer card source (D2')      -> paper-derived only; KYKT-job-derived
                                       deferred to cycle 010.
3. TEACHER_AUDIENCE_PROFILE (D3')  -> populated 2026-05-04 by user input.
                                       Earlier snapshot text claimed
                                       three sub-fields remained empty;
                                       this was a stale read. The user
                                       2026-05-04 input populated all 6
                                       fields explicitly: Research Taste
                                       = 科研的训练 / 写作技巧 /
                                       创新范式 / 讲好故事; Prior
                                       Expectations = "老师不知道我们
                                       要做 Dream" -> cold start; Demo
                                       Precedent -> first impression
                                       (implied by Prior Expectations);
                                       Previously Praised = "老师对 3R
                                       方向都没啥具体贬褒" -> no specific
                                       3R praise; Previously Criticized
                                       = same statement -> no specific
                                       3R criticism; Hard Constraints =
                                       no constraints stated. Profile
                                       is fully populated; D3' resolved;
                                       D3 (first demo target choice)
                                       still deferred per DEC-20260504-
                                       002 because cycle 010 case-card
                                       data for Memory + Permanence is
                                       still pending (agent
                                       recommendation per DEC-20260504-
                                       005 cycle 010 launch memo).
4. Cycle 009 launch authorization  -> go. CASE-20260504-CRITIC-01 is the
   (D4')                             first card per cycle 008 D1; drafted
                                     2026-05-04, paper-derived per D2'.
```

Cycle 010 launch decisions (locked 2026-05-04 from user message "1 a 吧 / 2 并行是不是好点儿 / 3 我觉得你决定就好 / 4 这个啥意思" plus agent confirmation; recorded in `decisions/DEC-20260504-005-cycle-010-launch.md`):

```text
1. v2 cost-typed route_regret (E1)  -> adopt. Contract bumped to v2;
                                       capability_match adds
                                       cost_normalized axis; alpha = 0.5
                                       initial (inferred). v1 prose
                                       preserved as Superseded per
                                       Discipline rule 5. CASE-COMPOSER-
                                       03 v2 row promoted to canonical
                                       recommendation; v1 row preserved.
                                       Memo: DEC-20260504-004.
2. Cycle 010 ordering (E2)          -> Memory + Permanence in parallel.
                                       Cross-pair pattern from cycle 009
                                       (CRITIC-02 <-> COMPOSER-01) reused;
                                       CR-2 binding closed via in-cycle
                                       cross-pair (Permanence publishes
                                       suppress_static_write, Memory
                                       consumes; both drafted in cycle
                                       010). Forward-reference null
                                       protocol from CRITIC-03 covers
                                       any timing gap.
3. D3 first demo target (E3)        -> agent decision: continued deferral
                                       to cycle 010 closeout. Two
                                       deferral conditions per
                                       DEC-20260504-002 are now both met
                                       (audience profile populated +
                                       cycle 009 case-card data exists),
                                       but Memory + Permanence still have
                                       no L2 evidence; picking now would
                                       violate Discipline rule 5
                                       (Honesty Override) by selecting on
                                       partial portfolio coverage. Re-
                                       surface at cycle 010 closeout when
                                       all 4 finalists have L2 cards.
4. D3' sub-field correction (E4)    -> stale snapshot text saying "three
                                       sub-fields remain empty" was
                                       wrong; profile is fully populated
                                       (see corrected entry in section 3
                                       above). User asked "这个啥意思"
                                       triggered the correction.
```

Cycle 011 launch decisions (locked 2026-05-05 from user message "你给我决定吧，（1）（2）（3）" delegating to agent; recorded in `decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-target.md`):

```text
1. D3 first teacher demo target (1)  -> Critic (Geometry Critic /
                                        System-2 3R, SPEC-20260503-001).
                                        Picked on five-axis comparison
                                        (surprise hook strength,
                                        mechanism legibility for
                                        cold-start audience, connection
                                        to Dream3R thesis, L2 portfolio
                                        depth, demo failure-mode
                                        robustness, "looks like paper X"
                                        collapse risk). Storyboard:
                                        STORY-20260505-001-critic.md
                                        drafted; status = draft only;
                                        no `approved-for-showing`
                                        granted by DEC-001. D3 = "first
                                        demo target", not retiring of
                                        other finalists; DEC-20260504-
                                        002 still in force.
2. Cycle 011 scope (2)               -> G4 (CR-2 partial on synthetic
                                        identity-validation clip) +
                                        G5 (forward-reference null
                                        protocol formalization) closure
                                        primary; Critic demo storyboard
                                        draft secondary. G6 + G2 +
                                        KYKT-derived Composer card +
                                        L3 prototype + paper writing
                                        all explicitly deferred.
3. v2 -> v3 candidates (3)           -> v2 -> v2.1 additive revision:
                                        forward-reference null protocol
                                        formalized as a contract-pinned
                                        subsection. The other two
                                        candidates (8x8 grid partition
                                        for Permanence regions;
                                        identity_consistency threshold
                                        pinning at ~0.7) deferred and
                                        not promoted; rationale per
                                        DEC-001 (3): grid partition is
                                        Permanence implementation
                                        detail, threshold pinning needs
                                        measured anchors that don't yet
                                        exist.
4. New (4) blocked items unchanged from cycle 010 closeout (final
   thesis selection / reproduction / training / checkpoint download /
   KYKT navigation change / Codex frontend implementation / reusable
   Codex skill packaging / discarding any non-finalist track /
   declaring teacher-demo readiness; demo SHOWING also blocked).
```

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness, **showing any of the 4 demo storyboards (Critic / Memory / Permanence / Composer; all remain `draft`; promotion to `approved-for-showing` requires a separate DEC per finalist)**.

Cycle 013 launch decisions (locked 2026-05-05 from user message "好了，请你做实际的研究部署吧，无论是准备工作还是调研和资料搜集等" + clarification "Phase 2 准备 + 资料调研 (推荐)" delegating to agent; recorded in `decisions/DEC-20260505-003-cycle-013-launch.md`):

```text
1. Cycle 013 scope (delegated)        -> Phase 2 preparation + research
                                        mining. Three locked sub-passes:
                                        S2 source-mining cycle-013 pass
                                        (4 finalist x 3 axes coverage
                                        gaps); S3 paper related-work
                                        prose draft (replace skeleton
                                        bullets with prose in Sections
                                        1-7; draft Sections 8-9 as
                                        prose); S4 L3 prerequisites
                                        briefs per finalist (4 markdown
                                        files under experiments/).
                                        Cycle 013 explicitly excluded:
                                        L3 prototype, checkpoint
                                        download, KYKT runner log
                                        access, model touching,
                                        retroactive edits to prior
                                        cycle artifacts, contract
                                        revision.
2. v2.1 -> v2.2 candidates (delegated) -> NO revision in cycle 013.
                                        No new candidate surfaced.
                                        Both cycle-011 deferred
                                        candidates (8x8 grid partition;
                                        identity_consistency threshold
                                        pinning) remain deferred.
                                        VGGT capability-card gap
                                        surfaced by cycle-013 mining
                                        is per-card, not contract.
3. D3 first demo target (carried)     -> unchanged; Critic per cycle
                                        011 DEC-20260505-001. No
                                        cycle-013 reconsideration.
4. Blocked items                       -> unchanged from cycle 012
                                        closeout. "Showing any of
                                        the 4 demo storyboards"
                                        unchanged. L3 / clone /
                                        download / install / run / KYKT
                                        navigation change / frontend /
                                        thesis finalization / training
                                        all still gated.
```

Cycle 013 G2 status update:

```text
Before cycle 013: G2 = inferred-with-real-inventory-anchor (per cycle
                       012 COMPOSER-04 KYKT-metadata-derived card).
After  cycle 013: unchanged. EXP-20260505-004 inventories the closure
                  path (multi-regime workload; 3+ backbones; measured
                  route_regret) but cycle 013 did NOT execute it.
Closure remains gated on L3 prototype OR KYKT runner log access; both
require separate user authorization.
```

New tracking goal G7 introduced cycle 013:

```text
G7 (paper-related-work-prose-readiness):
  Status after cycle 013: inferred-with-prose-draft-anchor.
  Sections 1-7 of literature/PAPER_RELATED_WORK_SKELETON.md are prose;
  Sections 8-9 are drafted as prose. The cycle-009 case-card-gate that
  blocked prose drafting is lifted (cleared by cycle 010-012 case-card
  + storyboard portfolio).
  Closure: full Phase-2 paper writing (intro / methods / results /
  discussion). Gated on user direction on venue / length / scope.
```

Cycle 012 launch decisions (locked 2026-05-05 from user message "你给我规划吧，然后弄完后告诉我我们的工作做了哪些？" delegating to agent; recorded in `decisions/DEC-20260505-002-cycle-012-launch.md`):

```text
1. Storyboard reviewer pass (1)      -> NOT done in cycle 012. Deferred
                                        to demo-show-authorization
                                        moment when functional-vs-
                                        placeholder tradeoffs become
                                        concrete. Critic storyboard +
                                        the 3 new storyboards all
                                        remain `draft` unchanged in
                                        future agent passes (no silent
                                        revisions).
2. Cycle 012 scope (2)               -> (c) KYKT-metadata-derived
                                        COMPOSER-04 + (e) 3 finalist
                                        demo storyboards (Memory +
                                        Permanence + Composer; all
                                        markdown only; all `draft`).
                                        Options (a) close G6 / (b)
                                        close G2 / (d) request demo
                                        show authorization / (f) start
                                        paper related-work writing all
                                        deferred (gated or premature).
3. v2.1 -> v2.2 candidates (3)       -> NO revision. Both deferred
                                        candidates from cycle 011
                                        (8x8 grid partition for
                                        Permanence regions;
                                        identity_consistency threshold
                                        pinning at ~0.7) remain
                                        deferred. COMPOSER-04 fits
                                        cleanly into existing v2
                                        schema; no new sub-axis needed.
4. (4) blocked items unchanged from cycle 011 closeout; "showing the
   Critic demo" extended to "showing any of the 4 demo storyboards"
   (all 4 finalists now have draft storyboards as of cycle 012).
```

Cycle 012 G2 status update:

```text
Before cycle 012: G2 = inferred (tau_spread = 0.05 paper-derived;
                                 capability_match paper-derived).
After cycle 012:  G2 = inferred-with-real-inventory-anchor
                       (capability_match values now KYKT-metadata-
                        derived per COMPOSER-04; tau_spread itself
                        still inferred; closure still requires
                        measured route_regret; gated).
G2 NOT closed. Closure remains gated on L3 prototype OR KYKT runner
log access.
```

Cycle 015 launch decisions (locked 2026-05-05 from user message "授权 Critic L3 窄域 pilot" delegating cycle 015 entry to the Critic L3 pilot scope; recorded in `decisions/DEC-20260505-005-cycle-015-launch-critic-l3-pilot.md`):

```text
1. Cycle 015 entry (D5)              -> authorize Critic L3 pilot scope
                                        per planning/L3_PILOT_SELECTION.md
                                        "Recommended first-pilot scope" +
                                        EXP-20260505-001 prerequisite
                                        inventory. Allowed: clone Test3R
                                        + CTRL + DUSt3R + MASt3R; download
                                        required checkpoints; install
                                        minimum env on a single box; run
                                        one smoke loop on one hard-case
                                        input; emit one JSONL log;
                                        write thin wrapper
                                        dream_critic_loop.py + hand-
                                        derived capability_match YAML.
2. Per-step micro gates (D5')        -> 5 gates, each a separate
                                        user go in active conversation:
                                        G_clone, G_install, G_download,
                                        G_run, G_log_use. DEC-005 alone
                                        does NOT authorize any of these
                                        steps; each surfaces individually
                                        with proposed path / repos /
                                        checkpoints / hard-case input
                                        before execution.
3. v2.1 -> v2.2 candidates (D5'')    -> NO revision in cycle 015. Both
                                        cycle-011 deferred candidates
                                        (8x8 grid partition; identity_
                                        consistency threshold pinning)
                                        remain deferred. VGGT capability-
                                        card gap remains per-card
                                        (CASE-20260505-COMPOSER-05),
                                        not contract-level.
4. Composer second-pilot precondition (D5''') -> unchanged from
                                        L3_PILOT_SELECTION.md: VGGT row
                                        must be included or explicitly
                                        excluded with reason before any
                                        Composer route_regret sweep is
                                        frozen. Cycle 015 does not act on
                                        this; Composer pilot is OUT of
                                        cycle 015 scope.
5. Blocked items unchanged from cycle 014 closeout, with one tightening:
   the cycle-015 authorization is Critic-L3-only. Memory / Permanence /
   Composer L3 prototypes remain blocked. Final thesis selection,
   reproduction (in the sense of paper-result re-runs), training,
   KYKT navigation change, frontend implementation, reusable Codex
   skill packaging, retiring any non-finalist track, declaring
   teacher-demo readiness, and showing any of the 4 demo storyboards
   all remain blocked.
```

Cycle 015 G2 / G6 / G7 status update at launch:

```text
G2 (route_regret closure):  unchanged. inferred-with-real-inventory-
                            anchor (cycle 012 anchor). Critic smoke
                            does NOT close G2; G2 closure remains
                            gated on multi-regime measured
                            route_regret OR KYKT runner log access.
G6 (memory governance):     unchanged.
G7 (paper related-work
     prose readiness):      unchanged. inferred-with-blueprint-anchor
                            (cycle 014 anchor). Closure still gated
                            on user direction on venue / length /
                            scope of Phase 2 paper.
```

Cycle 016 mainline redirect (locked 2026-05-06 from user message "所以现在我们做的和新模型有啥关系？或者说什么时候能开始推进主线了？" + selection of "B. Architecture-first" from a 3-option strategic question; recorded in `decisions/DEC-20260506-001-mainline-architecture-first.md`):

```text
1. Mainline definition (D6)         -> Dream's mainline is
                                       **architecture-first**: design a
                                       new 3R architecture (transformer
                                       / SSM / state-space / hybrid) as
                                       a markdown spec + ablation plan
                                       + comparator map. NOT framework-
                                       first paper writing.

2. Cycle 015 posture (D6')          -> stays paused at S9 done. NOT
                                       closed. NOT abandoned. The L3
                                       infrastructure (test3r conda env
                                       on server; launch.py patch; F-002
                                       memory; 4 local shallow clones)
                                       is reusable as evidence anchor
                                       for the architecture spec's
                                       Critic-module section. G_run can
                                       be resumed later if the
                                       architecture spec needs measured
                                       evidence; until then, no G_run.

3. Paper Phase 2 blueprint (D6'')   -> demoted from primary output to
                                       SUPPORT artifact. Still useful
                                       (control-graph theory becomes
                                       the THEORY behind the
                                       architecture); but the
                                       architecture spec is the
                                       PRIMARY output of the project.

4. Past decisions still in force (D6''') -> DEC-20260501-004 (Dream3R
                                       candidate-not-final) and
                                       DEC-20260504-002 (no-all-in any
                                       single finalist) both still
                                       apply. Architecture spec must
                                       (a) remain a candidate that can
                                       be revised/replaced/merged, and
                                       (b) preserve all 4 finalist
                                       mechanisms as composable
                                       modules, not collapse into one.

5. Train-first (option C) NOT authorized. Architecture-first
   authorizes design + ablation planning, NOT training, NOT GPU
   ablation runs, NOT checkpoint creation. Train-first remains
   deferred / blocked.

6. Blocked items unchanged from prior cycles. Hard rules carried from
   AGENT_MASTER_PROMPT.md section 6 unchanged: no reproduction / no
   checkpoint download / no training / no KYKT navigation change / no
   frontend / no thesis finalization / no retiring of any non-finalist
   track / no demo storyboard promotion past `draft`.

7. Cycle 016 launch deferred. S1 of cycle 016 = this DEC + feedback
   memory + this snapshot block (done 2026-05-06). S2..S5 (architecture
   spec draft + ablation plan + comparator map + sync chain) deferred
   to next session per user "今日进度到此为止".
```

Cycle 016 G2 / G6 / G7 status update at redirect:

```text
G2 (route_regret closure):  unchanged. Architecture spec drafting does
                            NOT close G2 (still gated on measured
                            route_regret OR KYKT runner log access).
G6 (memory governance):     unchanged.
G7 (paper related-work
     prose readiness):      unchanged. Paper output is now SUPPORT,
                            not primary; G7 closure is no longer the
                            project's main milestone. The new primary
                            milestone is "Dream3R architecture spec
                            v1 draft" (call it G8 if/when formalized).
                            Whether G7 stays open vs is retired is a
                            separate cycle-016 decision.
```

## Update protocol (highest priority — always honor)

This file MUST be updated at every meaningful task transition. The transitions are:

1. **Start of a task pass**: set `Status` to `in_progress`; populate `Current task`, `Subtask board`, and `If interrupted, resume from`.
2. **Each subtask completion**: flip the row's `Status` to `done` immediately. Don't batch.
3. **Mid-task interrupt or failure**: leave `Status` as `in_progress` and the active subtask as the last non-`done` row. Add a brief "Why interrupted" note. The next session resumes from there.
4. **End of a task pass (clean)**: flip `Status` to `idle`, mark `Last completed task pass` with what just finished, clear obsolete subtasks.
5. **End of a task pass (blocked on user)**: set `Status` to `blocked`; surface the blocker in `Open user decisions`.

Updating `Last updated:` is not optional. If you update this file you stamp it.

This rule is part of the Guidance File Sync Rule chain (see `WORKFLOW_STATUS.md`). It runs **first** in that chain — TASK_SNAPSHOT.md updates before any other file in a sync pass, so an interrupted sync still leaves a valid resume pointer.

## Recent failure modes the next agent should NOT repeat

Captured here as a short list because they are easy to repeat and expensive to recover from. Full prose lives in `cycles/CYCLE-20260504-001.md` "Note On The Earlier 32 MB Failure" and `RESEARCH_STATE.md` "Note On The Earlier 32 MB Failure".

```text
F-001  32 MB request-limit failure (2026-05-04, prior session)
       cause:  cumulative context (multiple full-file Reads + edit history)
               in one window, NOT any single oversized file
       avoid:  Read with offset/limit; Grep with -n for precision; Edit
               (old_string/new_string) instead of Write (full rewrite);
               keep <=2 large state files in context simultaneously;
               cite already-drafted content from cycle log + SPINE files
               instead of re-deriving it

F-002  agent assumed local Windows execution for KYKT 3R model work
       (2026-05-05, cycle 015 S7 G_install pre-flight)
       cause:  agent probed local GPU/Python/conda and proposed creating
               a conda env on the local Windows box and pip-installing
               dust3r/mast3r/Test3R locally. Did NOT consult
               E:\kykt\WORK.md or
               E:\kykt\.omx\plans\kykt-app-backend-model-integration-plan.md,
               both of which document a canonical server-side topology:
               KYKT 3R models live on /hdd3/kykt26 on a remote server,
               reached via system ssh + scp. Existing runners
               (dust3r_runner.py, mast3r_runner.py, monst3r_runner.py,
               spann3r_runner.py, fast3r_runner.py) already provide envs
               + checkpoints for 4 of the 5 inventoried backbones.
               EXP-20260505-001 was paper-derived and did NOT anchor to
               the production runner inventory; the agent inherited the
               paper-derived framing without re-checking topology.
       avoid:  before any L3 / training / heavy-compute G_install /
               G_download / G_run gate proposal, read E:\kykt\WORK.md +
               E:\kykt\.omx\plans\kykt-app-backend-model-integration-plan.md
               first; default to SERVER-side execution; reuse existing
               runners before installing new ones; ask the user for
               SSH host / path when topology details are missing rather
               than improvising. Local Windows box = markdown + shallow
               code-reading clones + orchestration + result inspection
               only; not the model execution target.
```

Add new entries here as new failure modes are discovered. Do not delete entries; supersede via a "superseded by F-NNN" note (Discipline rule 5 Honesty Override).

## Working rules to avoid F-001 (anti-32MB request-limit)

These rules apply to every agent operating in this workspace, including humans driving an agent. Violating them is the most common cause of multi-edit pass failures.

1. Do not re-Read a file already read earlier in the same conversation. The content is still in context. Cite line numbers from the prior Read; if a slice is needed, use Read with `offset`+`limit`.
2. For lookup, prefer `Grep -n` with `-C` / `-A` / `-B` for context. Reserve full Read for files under ~300 lines OR when the whole file is genuinely needed.
3. Prefer `Edit` (old_string / new_string, diff-only payload) over `Write` (full file payload) for any file that already exists. Use `Write` only for new files.
4. Cap "large" files (>300 lines OR >20 KB) in active context at <=2 simultaneously. When starting a new sync sub-pass, treat earlier large files as evicted; rely on `TASK_SNAPSHOT.md` and the most recent cycle log summary instead of re-Reading.
5. If a single Edit's old_string / new_string would exceed ~200 lines, split into multiple smaller Edits anchored at stable section headers, not in the middle of paragraphs.
6. Sync `TASK_SNAPSHOT.md` FIRST in any Guidance File Sync Rule chain pass, so an interrupt mid-pass still leaves a valid resume pointer (this is the rule already restated in `WORKFLOW_STATUS.md` "Guidance File Sync Rule").
7. Do not run multi-file Read fan-outs in parallel just to "have everything"; pull files in only when the next concrete edit demands them.
8. If a single tool call returns "Request too large", do NOT retry with the same payload. Switch to: smaller Read window, narrower Grep, or split Edit. Record the trigger in this section's F-NNN list if it represents a new failure mode.

These rules are mandatory-equivalent to the Hard rules below, since violating them tends to lose a session's worth of work.

## Hard rules (carried from AGENT_MASTER_PROMPT.md, restated for safety)

1. No reproduction. No checkpoint download. No training. No KYKT navigation change. No frontend implementation. No thesis finalization. No retiring of any non-finalist track. None of these without explicit user approval **in the active conversation**.
2. Apply Surgical Edits (Discipline rule 3): every changed line traces to a user request, decision memo, or Sync Rule trigger.
3. Apply Honesty Override (Discipline rule 5): every mechanism claim carries an evidence label; user approval cannot be invented.
4. ID conventions are fixed: `SPEC-YYYYMMDD-NNN`, `DEC-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`. Use the current session date prefix for new artifacts.
5. Sentence case headers. No em-dashes.
