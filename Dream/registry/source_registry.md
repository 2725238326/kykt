# Source Registry

Last updated: 2026-05-02

Status: seeded from Phase 1 source map. This file is the lightweight index; detailed notes stay in `FRONTIER_SOURCE_MAP.md` and cycle logs.

## Schema

```text
source_id:
title:
url:
year:
track:
evidence:
mechanism_one_liner:
readiness:
linked_units:
```

## Seed Sources

| ID | Title | URL | Year | Track | Evidence | Mechanism | Readiness | Linked units |
|---|---|---:|---:|---|---|---|---|---|
| SRC-2024-001 | DUSt3R | https://github.com/naver/dust3r | 2024 | direct_3r | paper, code, checkpoints, demo | pose-free pointmap reconstruction | P0 baseline | RU-002, RU-014 |
| SRC-2024-002 | MASt3R | https://github.com/naver/mast3r | 2024 | direct_3r | paper, code, checkpoints, demo | 3D-grounded matching and sparse global alignment | P1 baseline | RU-002 |
| SRC-2024-003 | MonST3R | https://github.com/Junyi42/monst3r | 2024 | direct_3r | paper, code, demo | dynamic-video pointmap reconstruction | P1 dynamic | RU-013 |
| SRC-2024-011 | Spann3R | https://github.com/HengyiWang/spann3r | 2024 | direct_3r | paper, code | spatial memory for global pointmap prediction | P1 memory comparator | RU-001, RU-004, RU-014, RU-015 |
| SRC-2025-001 | Fast3R | https://github.com/facebookresearch/fast3r | 2025 | direct_3r | paper, code, checkpoint, demo | many images in one forward pass | P1 scale | RU-002, RU-014 |
| SRC-2025-002 | CUT3R | https://github.com/CUT3R/CUT3R | 2025 | direct_3r | paper, code, checkpoints, demo | persistent-state continuous 3D perception | P2 architecture baseline | RU-001, RU-014 |
| SRC-2025-003 | Point3R | https://github.com/YkiWu/Point3R | 2025 | direct_3r | paper, code, checkpoint | explicit spatial pointer memory | P2 comparator | RU-004, RU-014 |
| SRC-2026-001 | STream3R | https://github.com/NIRVANALAN/STream3R | 2026 | direct_3r | paper, code, app, inference path | causal transformer and stream session for 3R | P2 comparator | RU-010, RU-014 |
| SRC-2025-004 | TTT3R | https://github.com/Inception3D/TTT3R | 2025 | architecture_transfer | paper, code, demo command | test-time update rule for CUT3R | P2 add-on | RU-003, RU-011, RU-012 |
| SRC-2024-004 | Splatt3R | https://github.com/btsmart/splatt3r | 2024 | demo_enabler | paper, code, checkpoint, Gradio | uncalibrated image pairs to 3D Gaussians | P0 visual | RU-008 |
| SRC-2024-005 | InstantSplat | https://github.com/NVlabs/InstantSplat | 2024 | demo_enabler | paper, code, scripts | sparse-view SfM-free Gaussian Splatting | P0/P1 visual | RU-008 |
| SRC-2025-005 | MV-DUSt3R+ | https://github.com/facebookresearch/mvdust3r | 2025 | direct_3r | paper, code, checkpoints, Gradio | sparse-view pose-free RGB reconstruction | P1 visual/geometry | RU-008 |
| SRC-2025-006 | NoPoSplat | https://github.com/cvg/NoPoSplat | 2025 | demo_enabler | paper, code, checkpoints | sparse unposed images to Gaussians | P2 comparator | RU-008 |
| SRC-2023-001 | Mamba | https://arxiv.org/abs/2312.00752 | 2023 | architecture_transfer | paper, code known | selective state space sequence modeling | mechanism | RU-001 |
| SRC-2024-006 | Mamba-2 / SSD | https://arxiv.org/abs/2405.21060 | 2024 | architecture_transfer | paper, code | state space duality and efficient state layer | mechanism | RU-001, RU-010 |
| SRC-2024-007 | VMamba | https://arxiv.org/abs/2401.10166 | 2024 | architecture_transfer | paper, code | 2D selective scan routes | mechanism | RU-009 |
| SRC-2024-008 | MambaOut | https://arxiv.org/abs/2405.07992 | 2024 | negative_control | paper, code | SSM is not always needed for vision | caution | RU-001, RU-010 |
| SRC-2025-007 | Test3R | https://arxiv.org/abs/2506.13750 | 2025 | architecture_transfer | paper, code | test-time geometric consistency | P2/P3 | RU-003, RU-011 |
| SRC-2025-008 | CTRL | https://arxiv.org/abs/2502.03492 | 2025 | architecture_transfer | paper, code | critic-revision through RL-trained critic | mechanism | RU-011 |
| SRC-2025-009 | SEAL | https://arxiv.org/abs/2506.10943 | 2025 | architecture_transfer | paper, code | self-edit driven adaptation | mechanism | RU-012 |
| SRC-2025-010 | POMATO | https://arxiv.org/abs/2504.05692 | 2025 | direct_3r | paper | pointmap matching plus temporal motion | comparator | RU-013 |
| SRC-2025-011 | D^2USt3R | https://arxiv.org/abs/2504.06264 | 2025 | direct_3r | paper | 4D pointmaps for dynamic scenes | comparator | RU-013 |
| SRC-2025-012 | LONG3R | https://arxiv.org/abs/2507.18255 | 2025 | direct_3r | paper, project | memory gating and 3D spatio-temporal memory for long sequences | core comparator | RU-001, RU-014, RU-015 |
| SRC-2026-002 | LoGeR | https://arxiv.org/abs/2603.03269 | 2026 | direct_3r | paper, project | chunked long-context reconstruction with TTT global memory and SWA local memory | core comparator | RU-010, RU-014, RU-015 |
| SRC-2026-003 | Mem3R | https://arxiv.org/abs/2604.07279 | 2026 | direct_3r | paper, project | hybrid memory decoupling camera tracking from geometric mapping | core comparator | RU-004, RU-012, RU-015 |
| SRC-2026-004 | PAS3R | https://arxiv.org/abs/2603.21436 | 2026 | direct_3r | paper | pose-adaptive streaming state update | core comparator | RU-001, RU-015 |
| SRC-2026-005 | FILT3R | https://arxiv.org/abs/2603.18493 | 2026 | direct_3r | paper, code promised | Kalman-style latent filtering for streaming 3R | mechanism | RU-001, RU-015 |
| SRC-2026-006 | LongStream | https://arxiv.org/abs/2602.13172 | 2026 | direct_3r | paper, project | gauge-decoupled streaming visual geometry and cache refresh | comparator | RU-010, RU-014, RU-015 |
| SRC-2026-007 | OVGGT | https://arxiv.org/abs/2603.05959 | 2026 | visual_geometry | paper, project, code | constant-budget cache compression and dynamic anchor protection | mechanism | RU-010, RU-014, RU-015 |
| SRC-2026-008 | RayMap3R | https://raymap3r.github.io/ | 2026 | dynamic_3r | project, code claimed | inference-time RayMap for dynamic streaming reconstruction | comparator, verify before reproduction | RU-013, RU-015 |
| SRC-2024-009 | MASt3R-SfM | https://arxiv.org/abs/2409.19152 | 2024 | direct_3r | paper, code via MASt3R ecosystem | matching, retrieval, and global SfM alignment | comparator | RU-002, RU-015 |
| SRC-2024-010 | SLAM3R | https://arxiv.org/abs/2412.09401 | 2024 | direct_3r | paper, code | sliding-window dense SLAM using pointmap prediction and registration | comparator | RU-014, RU-015 |
| SRC-2025-013 | Easi3R | https://arxiv.org/abs/2503.24391 | 2025 | dynamic_3r | paper, project | training-free dynamic adaptation / motion separation | comparator | RU-013, RU-015 |
| SRC-2025-014 | G-CUT3R | https://arxiv.org/abs/2508.11379 | 2025 | guided_3r | paper | guided CUT3R with depth / calibration / pose priors | comparator | RU-002, RU-015 |
