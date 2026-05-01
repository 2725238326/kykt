# Branch Comparison Matrix

Last updated: 2026-05-02

Status: scaffold, not scored yet.

## Purpose

Compare multiple research branches before betting on one direction.

## Matrix

| Branch | Failure modes addressed | Closest competitors | Mechanism ingredients | Possible compositions | Novelty gap | Smallest evidence path | Teacher-facing demo | Engineering cost | Top-conference risk | KYKT support path | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Executive Memory / State Governance | F1, F2, F3 | Spann3R, CUT3R, Point3R, LoGeR, Mem3R, PAS3R, FILT3R, OVGGT | state update, external memory, anchors, cache, critic triggers | Kalman update + anchor cache + geometry critic | TBD | proxy policy simulation | memory action timeline | medium | may look incremental | research lane / advisor | compare |
| Geometry Critic / System-2 3R | F3, F1 | Test3R, TTT3R, geometry consistency methods | critic, revision, adaptive compute, rerouting | critic + composer + memory write policy | TBD | non-learned consistency report | model catches and fixes error | low-medium | may be diagnostics only | advisor / report | compare |
| Dynamic Object Permanence / 4D Memory | F2, F1 | MonST3R, POMATO, D^2USt3R, Easi3R, RayMap3R, 4DGS | dynamic pointmaps, object memory, static/dynamic split, 4D asset bridge | object permanence + 4DGS + memory governance | TBD | dynamic failure-case taxonomy | moving object remains consistent | medium-high | may become graphics demo | dynamic sample lane | compare |
| Cross-Modal / Event-Augmented 3R | F5, F2, F3 | EAG3R, Event-3DGS, event depth/reconstruction | event/RGB fusion, blur-free temporal signal, sensor priors | event signal + geometry critic + dynamic branch | TBD | dataset-only analysis first | blurred RGB vs event-stable reconstruction | medium-high | hardware/data burden | research lane only first | compare |
| 3R Composer / Unified Model Ecology | F6, F3, F1 | MASt3R-SfM, Fast3R, SLAM3R, MV-DUSt3R+, Splatt3R | capability cards, model routing, unified contracts, evidence reports | composer + critic + benchmark | TBD | model capability matrix | system chooses model by failure mode | low-medium | may be system not paper | strong KYKT fit | compare |
| Active Spatial Perception / RL-3R | F4, F3, F2 | NBV, active perception, VLA / robotics systems | information gain, uncertainty map, action policy, RL/planning | active view + memory uncertainty + critic | TBD | simulation/design study first | model asks camera to move | high | sim2real and scope | future robotics lane | compare |

## Scoring Fields To Fill

Use 1-5 scores:

- novelty after comparator check
- paper crispness
- evidence feasibility
- demo surprise
- engineering feasibility
- KYKT fit
- risk control

## Next Action

Fill this matrix using:

- `FRONTIER_SOURCE_MAP.md`
- `RESEARCH_UNIT_BANK.md`
- `DREAM3R_THESIS_STRESS_TEST.md`
- `MULTI_TRACK_RESEARCH_CANVAS.md`

Then select 2-3 finalist branches for deeper mechanism specs.
