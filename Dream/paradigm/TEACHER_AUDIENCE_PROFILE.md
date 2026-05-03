# Teacher Audience Profile

Last updated: 2026-05-04 (cycle 008.5; placeholder; awaits user input)

Status: **placeholder**. Fields below are intentionally blank. The agent must not fill them without explicit user input. Until populated, decision D3 (first teacher demo target) and `templates/demo_storyboard.md` audience-assumption fields stay open.

## Purpose

The four finalist specs (Critic / Memory / Permanence / Composer) each describe a possible teacher demo. Picking which one to show first depends on the teacher's research taste, prior expectations, and demo precedent. None of those are knowable to the agent.

This file is the user's input surface. When populated, it gates D3 (first teacher demo target). Until then, D3 is deferred per `decisions/DEC-20260504-002-no-all-in-on-single-finalist.md`.

## How To Use This File

1. The user fills the fields below.
2. The agent reads them when next asked to plan a demo or storyboard.
3. The agent does NOT invent values for blank fields.
4. If the agent must reason about a demo path before this file is populated, the agent surfaces a question to the user instead of inferring.
5. When the file changes substantively, append a dated note to the change log at the end rather than silently overwriting.

## Fields For The User To Populate

### Research Taste

```text
[blank for user]
```

Examples of what to capture (the user picks the framing that fits):

- theory-leaning: prefers principled formulations and crisp claims
- system-leaning: prefers measurable engineering and end-to-end demos
- visual-leaning: prefers visible artifacts, pictures, timelines

A teacher may be more than one. Order them by weight if so.

### Prior Expectations On This Work

```text
[blank for user]
```

Examples:

- cold start: teacher has not seen Dream content; first impression matters most
- known direction: teacher has seen prior framing (e.g. Dream3R / GEM-3R) and expects continuation
- skeptical baseline: teacher has expressed concern about a specific axis

### Demo Precedent

```text
[blank for user]
```

Examples:

- first impression: this is the first demo of Dream the teacher will see
- reinforcement: teacher has seen earlier work; this demo continues a thread
- comparison: teacher will see this demo alongside another lab's work

### What The Teacher Has Previously Praised In Similar Work

```text
[blank for user]
```

Examples (the user fills with specifics, not categories):

- particular paper claims the teacher cited approvingly
- particular demo styles the teacher found compelling
- particular comparators the teacher views as benchmarks

### What The Teacher Has Previously Criticized In Similar Work

```text
[blank for user]
```

Examples:

- common research moves the teacher rejects (e.g. "module-stacking", "buzzword combinations")
- positioning failures the teacher cites
- demo failures the teacher would notice

### Hard Constraints

```text
[blank for user]
```

Examples:

- timing: how long the demo is allowed to run
- presentation surface: laptop screen, projector, shared notebook, etc.
- live vs offline: must everything run offline; can the demo include live inference

## Agent Behavior While This File Is Empty

- Do NOT pick a demo target.
- Do NOT fill demo storyboard audience fields by inference.
- Do NOT recommend "Critic first" / "Memory first" / "Permanence first" / "Composer first" in user-facing summaries based on demo impact (technical sequencing per cycle 008 D1 still applies; that is execution order, not demo precedence).
- DO surface this file as an open dependency when the user asks about demo planning.
- DO continue with case cards in cycle 009 regardless; case cards are independent of demo authorization.

## Change Log

- 2026-05-04 (cycle 008.5): file created as placeholder per `handoff/SESSION-HANDOFF-20260504-001-cycle-008-closeout-and-cycle-009-prep.md` Task 11. No content. Awaiting user input.

## Companion Files

- `decisions/DEC-20260504-002-no-all-in-on-single-finalist.md` — the decision deferring D3 until this file is populated and case-card data lands
- `templates/demo_storyboard.md` — references this file as the audience profile pointer
- `WORKFLOW_STATUS.md` — Recommended Next User Decision lists population of this file as a still-open user task
