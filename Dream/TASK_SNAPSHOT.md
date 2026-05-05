# Dream Task Snapshot

Last updated: 2026-05-06 (cycle 015 S9 done: G_download was degenerate — Test3R hardcodes HF id `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` but huggingface.co blocked from server; user chose patch-launch.py path; one-line `sed -i` patch on /hdd3/kykt26/code/Test3R/eval/mv_recon/launch.py:103 to point at existing /hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth (2.2 GB; pre-existing for dust3r runner); `.cycle015.bak` backup kept; verified `AsymmetricCroCo3DStereo.from_pretrained(local_path)` loads 571.2M params with all keys matched, CUDA transfer OK; no HF network call; G_run / G_log_use still required as separate per-step gates; v2.1 unchanged; G2/G6/G7 unchanged; F-002 + memory persisted)

Status: **in_progress** (cycle 015 S10 next: surface G_run)

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
task_id:    cycle-015
phase:      Phase 2 execution prep (markdown-only launch + sync; per-step execution gated)
cycle:      015 (Critic L3 pilot scope authorized; clone / download / install / run still gated)
status:     in_progress (S1..S2 done; S3..S5 active; S6+ gated on G_clone / G_install / G_download / G_run / G_log_use)
```

One-line description:

```text
User selected "授权 Critic L3 窄域 pilot" as the cycle 015 entry, in
response to the agent's recommendation surfaced after cycle 014 closeout
in cycles/CYCLE-20260505-005.md "Cycle 015 launch packet (deferred to
user)". This satisfied the open user decision phrased in
planning/L3_PILOT_SELECTION.md line 167-171. DEC-20260505-005 records the
scope-level authorization (clone Test3R + CTRL + DUSt3R + MASt3R; download
required checkpoints; install minimum env; run one smoke loop on one hard
case; emit one JSONL log; write thin orchestration wrapper). Each
operational step (clone, install, checkpoint download, run, log commit)
is a per-step micro gate that requires a separate user go in the active
conversation; the DEC alone does NOT authorize them. Cycle 015 launch did
NOT authorize: full sweep, training, KYKT navigation change, frontend,
storyboard promotion, G2 closure claim, retroactive case-card edits,
system-level changes, silent upstream patches, teacher-demo readiness
claim, final thesis selection, retiring of any non-finalist track.
```

## Subtask board (active pass: cycle 015 launch + Critic L3 pilot scope authorization, 2026-05-05)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| S1 | Write DEC-20260505-005 (cycle 015 launch + Critic L3 pilot scope authorization; locks allowed / not-allowed; lists 5 per-step micro gates and stop conditions) | done | `decisions/DEC-20260505-005-cycle-015-launch-critic-l3-pilot.md` |
| S2 | Write cycles/CYCLE-20260505-006.md (cycle 015 cycle log; subtask board; gate roadmap) | done | `cycles/CYCLE-20260505-006.md` |
| S3 | Update this file FIRST per F-001 rule 6 (Status -> in_progress; cycle 015 board; resume pointer; cycle 015 launch decision under Open user decisions) | done | `TASK_SNAPSHOT.md` |
| S4 | Sync chain: WORKFLOW_STATUS.md + RESEARCH_STATE.md + INDEX.md + registry/decision_registry.md + AGENT_MASTER_PROMPT.md + README.md (cycle 015 pointer + DEC-20260505-005 row + last-updated stamp bumps) | done | sync chain |
| S5 | Surface G_clone micro gate to user (proposed pilot path under experiments/runs/cycle-015-critic-l3-pilot; proposed hard-case input candidate; GPU/env confirmation request) | done | this snapshot + user message |
| S6 | G_clone returned `go` (default path + 4 repos). Executed: pilot path created; `git clone --depth 1` Test3R + CTRL + DUSt3R + MASt3R; sizes + HEAD + license recorded in cycle log. License finding: CC BY-NC-SA 4.0 on Test3R/DUSt3R/MASt3R + Apache 2.0 on critic-rl; research use OK, NonCommercial constraint recorded for future commercial-path DEC. NO install / NO checkpoint / NO run took place. | done | `experiments/runs/cycle-015-critic-l3-pilot/` + `cycles/CYCLE-20260505-006.md` |
| S7 | Surface G_install micro gate (revised, server-side framing). S7 corrigendum + F-002 + feedback memory persisted; server coords resolved by reading ssh_runner.py:22-44; revised proposal: install only Test3R server-side; reuse DUSt3R / MASt3R / MonST3R / Spann3R / Fast3R envs already on server. User reply: `Go (server-side install Test3R only)`. | done | this snapshot + user message + `cycles/CYCLE-20260505-006.md` |
| S8 | G_install (revised) returned `go`. Initial server `git clone` failed (github.com unreachable from KYKT-UI: GnuTLS recv error / curl timeout); user redirected to scp local clone. Executed: scp local Test3R (18 MB, HEAD a2eb94b) -> /hdd3/kykt26/code/Test3R; verified size + HEAD match. conda env `test3r` created (Python 3.11.15); first conda-style pytorch install hit MKL `iJIT_NotifyEvent` symbol error (defaults channel mkl 2025.0.0 vs older pytorch build); within-scope fixup: dropped conda pytorch + pip-installed torch 2.5.1+cu121 from `https://download.pytorch.org/whl/cu121` (mirrors dust3r env approach on same server); torch sanity passed (4x TITAN RTX 24GB; CUDA 12.1). pip install -r Test3R/requirements.txt completed; all 11 direct deps import cleanly. EXP-001 inferred numbers upgraded to `measured` for GPU class / clone disk / install wall-clock. NO checkpoint download / NO run / NO KYKT runner change. | done | server `/hdd3/kykt26/code/Test3R/` + `/home/kykt26/.conda/envs/test3r/` + `cycles/CYCLE-20260505-006.md` |
| S9 | G_download surfaced; Test3R weights = HF id `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` but huggingface.co blocked. User chose `Patch launch.py 指向已有 ckpt`. Executed: backed up Test3R/eval/mv_recon/launch.py to .cycle015.bak; sed-patched line 103 to local path /hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth (2.2 GB pre-existing dust3r ckpt). Verified end-to-end model load: AsymmetricCroCo3DStereo.from_pretrained() loads 571.2M params, "All keys matched successfully", CUDA transfer to cuda:0 OK, no HF network call. G_download was degenerate (no actual download; ckpt reused). | done | server `/hdd3/kykt26/code/Test3R/eval/mv_recon/launch.py` + cycle log |
| S10 | Surface G_run micro gate to user (smoke loop scope: hard-case input, Critic conflict score, Composer-backed reroute, before/after delta log; agent will propose specific image-pair input candidate from CRITIC-01 failure taxonomy; thin wrapper dream_critic_loop.py drafted then user-reviewed before run) | in_progress | this snapshot + user message |
| S11+ | Reserved for execution sub-passes; G_log_use -> S11 | not started | TBD |

## Last completed task pass

```text
pass_name:        Cycle 014 launch + convergence pass (S1..S5 done;
                  markdown-only Phase 2 convergence and execution
                  selection)
date:             2026-05-05
trigger:          User message "继续" after the agent recommended cycle
                  014 scope: paper blueprint, VGGT Composer gap addendum,
                  and L3 pilot downselect.
files_modified:   TASK_SNAPSHOT.md (this file; subtask board + last
                  completed task pass + If interrupted resume from +
                  hard-rule date-prefix stale sentence corrected;
                  FIRST in sync chain per F-001 rule 6)
                  cycles/CYCLE-20260505-005.md (audit + closeout +
                  subtask board flips)
                  WORKFLOW_STATUS.md, RESEARCH_STATE.md, INDEX.md,
                  AGENT_MASTER_PROMPT.md, README.md,
                  registry/decision_registry.md (cycle 014 sync)
new_artifacts:    decisions/DEC-20260505-004-cycle-014-launch.md
                  cycles/CYCLE-20260505-005.md
                  literature/PAPER_PHASE2_BLUEPRINT.md
                  cases/CASE-20260505-COMPOSER-05.md
                  planning/L3_PILOT_SELECTION.md
in_place_edits:   guidance sync files only; no prior case cards, specs,
                  storyboards, contract, or experiment briefs rewritten
discipline:       Surgical Edits (new synthesis in new artifacts;
                  existing Composer cards not retroactively rewritten;
                  v2.1 contract unchanged) + Honesty Override (paper
                  blueprint is draft; VGGT rows inferred; L3 pilot is
                  recommendation only; no execution authorization;
                  G2/G6 unchanged; G7 not closed; storyboards draft).
budget_event:     None this pass. No large source maps or case-card
                  portfolios were re-read in full.

prior_pass_name:  Cycle 013 launch + closeout pass
prior_pass_date:  2026-05-05
prior_pass_files: TASK_SNAPSHOT.md, cycles/CYCLE-20260505-004.md,
                  sources/FRONTIER_SOURCE_MAP.md,
                  registry/source_registry.md,
                  literature/PAPER_RELATED_WORK_SKELETON.md,
                  experiments/EXP-20260505-001..004-l3-prerequisites-*.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).
2. Read cycles/CYCLE-20260505-006.md (cycle 015 in progress; S1..S2
   done; S3..S5 active) and decisions/DEC-20260505-005-cycle-015-
   launch-critic-l3-pilot.md (Critic L3 pilot scope locked; 10
   "not allowed" items; 5 per-step micro gates; 7 inherited + 3
   cycle-015-specific stop conditions; 7 acceptance criteria).
3. Cycle 015 is `in_progress`. Next action depends on which row in
   the Subtask board above is the latest non-`done` row.
4. If S3 is in_progress, finish the TASK_SNAPSHOT.md sync first per
   F-001 rule 6 (this file FIRST). Then S4 sync chain (WORKFLOW_
   STATUS / RESEARCH_STATE / INDEX / registry/decision_registry).
   Then S5 surface G_clone gate to user.
5. If S5 already surfaced and user has NOT replied "go", the cycle
   is gated. Do NOT clone / download / install / run anything; wait.
6. If user has replied "go" to G_clone, S6 starts: clone the 4
   primary repos (Test3R, CTRL, DUSt3R, MASt3R) at the agreed pilot
   path (proposed: experiments/runs/cycle-015-critic-l3-pilot/).
   Each subsequent step (G_install, G_download, G_run, G_log_use)
   is its own gate; do not chain them silently. If any micro gate
   returns "no" or "redirect", stop and surface to user; do not
   proceed to the next step.
7. All 4 finalist demo storyboards (STORY-20260505-001..004) remain
   markdown `draft` ONLY. Do NOT promote any to `approved-for-
   showing` without a separate DEC. Do NOT start any non-Critic L3
   pilot (Memory / Permanence / Composer L3) / training / KYKT
   navigation change / frontend implementation without explicit
   user approval per AGENT_MASTER_PROMPT.md section 6. The 4 L3
   prerequisite briefs under experiments/ are inventory; filing
   them was NOT authorization to execute them; only Critic
   (EXP-20260505-001) is in cycle-015 scope per DEC-20260505-005.
8. Honor F-001 working rules throughout: do not Read large files
   already cited in this snapshot; prefer Grep -n + Edit over full-
   file Read + Write; cap large files in active context at <=2
   simultaneously.
```

If this snapshot's Status is `idle`, cycle 015 is closed and the next live phase is cycle 016 (gated on cycle 015 closeout + a separate user direction).

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
