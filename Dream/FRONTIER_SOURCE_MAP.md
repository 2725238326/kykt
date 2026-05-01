# Frontier Source Map

Last updated: 2026-05-01

This is the first-pass source map for Dream Phase 1. It prioritizes primary sources: arXiv, CVF/OpenReview, official project pages, and official GitHub repositories.

Evidence labels:

- `paper`: paper exists
- `code`: official or credible code exists
- `demo`: project/demo page or runnable demo exists
- `speculation`: Dream inference, not proven by source

## Early Thesis Signal

The direct 3R frontier after DUSt3R is no longer only "better pointmaps." It is increasingly about:

```text
how to update, store, compress, query, and correct spatial state over time.
```

This supports Dream's architecture-first framing.

## Direct 3R / Pointmap Family

| Source | Year | Tag | Mechanism | Evidence | Dream relevance |
|---|---:|---|---|---|---|
| [DUSt3R](https://arxiv.org/abs/2312.14132) | 2023/2024 | direct_3r | Pose-free dense pointmap regression; global alignment for multi-view | paper, project/code known | Base paradigm; all follow-up ideas inherit pointmap output contract |
| [MASt3R](https://arxiv.org/abs/2406.09756) | 2024 | direct_3r | Adds 3D-grounded dense local features and fast reciprocal matching to DUSt3R | paper, code known | Strong static matching baseline; useful for composer lane |
| [MonST3R](https://arxiv.org/abs/2410.03825) | 2024/2025 | direct_3r | Extends pointmap estimation to dynamic video via per-timestep geometry fine-tuning | paper, project/code known | Dynamic geometry baseline already relevant to KYKT |
| [Spann3R](https://arxiv.org/abs/2408.16061) / [GitHub](https://github.com/HengyiWang/spann3r) | 2024/2025 | direct_3r | Spatial memory for global pointmap prediction without optimization-heavy alignment | paper, code | First memory-flavored 3R baseline in KYKT route |
| [Fast3R](https://arxiv.org/abs/2501.13928) | 2025 | direct_3r | Processes many images in one forward pass; avoids pairwise/global alignment bottleneck | paper, CVPR/code known | Speed/scaling baseline; good for composer and app demo |
| [MV-DUSt3R+](https://github.com/facebookresearch/mvdust3r) | 2025 | direct_3r | Sparse-view pose-free RGB-only reconstruction; supports NVS and relative pose estimation | code, Gradio, checkpoints | Good direct multi-view demo candidate; Linux/CUDA 12.4 path noted |
| [CUT3R](https://cut3r.github.io/) / [GitHub](https://github.com/CUT3R/CUT3R) | 2025 | direct_3r | Continuous updating transformer with persistent state for online pointmaps | paper, code, demo | Key stateful baseline; current KYKT env blocked but high conceptual value |
| [STream3R](https://arxiv.org/abs/2508.10893) / [GitHub](https://github.com/NIRVANALAN/STream3R) | 2025/2026 | direct_3r | Decoder-only / causal Transformer framing for sequential 3D reconstruction | paper, code, demo | Important comparator for any Mamba/linear-state claim |
| [Test3R](https://arxiv.org/abs/2506.13750) / [GitHub](https://github.com/nopQAQ/Test3R) | 2025 | architecture_transfer | Test-time geometric consistency objective over image triplets | paper, code | Best near-term path for "System-2 geometry self-check" demo |
| [TTT3R](https://arxiv.org/abs/2509.26645) | 2025/2026 | architecture_transfer | Frames streaming 3R as test-time training; confidence-derived memory update rate | paper, code page claimed | Strong bridge between memory and test-time adaptation |
| [LONG3R](https://arxiv.org/abs/2507.18255) | 2025 | direct_3r | Recurrent long-sequence 3R with memory gating and 3D spatio-temporal memory | paper, project page | Direct evidence that long-sequence memory is central |
| [Point3R](https://arxiv.org/abs/2507.02863) | 2025 | direct_3r | Explicit spatial pointer memory for streaming dense reconstruction | paper | Important for external geometric memory designs |
| [PAS3R](https://arxiv.org/abs/2603.21436) | 2026 | direct_3r | Pose-adaptive state update for long monocular streams | paper | Very close to Dream's proposed geometry-gated state update |
| [LoGeR](https://loger-project.github.io/) | 2026 | direct_3r | Hybrid memory for long-context geometric reconstruction | project page, paper link | Reinforces "context wall + data wall" framing |
| [Mem3R](https://lck666666.github.io/Mem3R/) | 2026 | direct_3r | Hybrid memory decoupling camera tracking and geometric mapping | project page | Relevant comparator for Dream memory designs |
| [RayMap3R](https://raymap3r.github.io/) | 2026 | direct_3r | Inference-time RayMap for dynamic streaming reconstruction | project page, code claimed | Useful for dynamic-memory and ray-based state ideas |
| [POMATO](https://arxiv.org/abs/2504.05692) | 2025 | direct_3r | Combines pointmap matching with temporal motion for dynamic 3D reconstruction | paper, code location claimed | Dynamic branch comparator; proves motion should be represented, not hidden in static alignment |
| [D^2USt3R](https://arxiv.org/abs/2504.06264) | 2025 | direct_3r | Regresses 4D pointmaps for dynamic scenes | paper | Strong evidence for pointmap-to-4D representation shift |

## Architecture Mechanisms

| Source | Year | Tag | Mechanism | Evidence | 3R translation |
|---|---:|---|---|---|---|
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023/2024 | architecture_transfer | Selective state spaces; input-dependent propagation/forgetting; linear scaling | paper, code known | Replace temporal attention with geometry-gated state update |
| [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) / [GitHub](https://github.com/state-spaces/mamba) | 2024 | architecture_transfer | State Space Duality; SSM layer closer to efficient attention with faster implementation | paper, code | Candidate state layer for 3R temporal memory; compare to causal attention |
| [Vision Mamba / Vim](https://arxiv.org/abs/2401.09417) | 2024 | architecture_transfer | Bidirectional Mamba visual backbone with memory/compute efficiency | paper, code | Candidate feature backbone for high-res 3R inputs |
| [VMamba](https://arxiv.org/abs/2401.10166) | 2024 | architecture_transfer | 2D selective scan over multiple routes for vision | paper, code | Spatial scan design for image/pointmap tokens |
| [QuadMamba](https://arxiv.org/abs/2410.06806) | 2024 | architecture_transfer | Quadtree-adaptive visual scan | paper, code claimed | Route policy for adaptive image/pointmap token ordering |
| [GroupMamba](https://arxiv.org/abs/2407.13772) | 2024 | architecture_transfer | Channel-grouped selective scan with modulation | paper, code claimed | Split 3R state into geometry / appearance / motion groups |
| [EfficientViM](https://arxiv.org/abs/2411.15241) | 2024/2025 | architecture_transfer | Hidden-state mixer and multi-stage hidden-state fusion | paper, code claimed | Suggests state fusion should happen inside hidden state, not only across tokens |
| [MLLA / MILA](https://arxiv.org/abs/2405.16605) | 2024 | architecture_transfer | Linear attention interpreted through Mamba-like gates | paper, code claimed | Alternative cheap global mixing layer for long-view 3R |
| [Infini-attention](https://arxiv.org/abs/2404.07143) | 2024 | architecture_transfer | Local attention plus compressive long-term memory in one block | paper | Useful abstraction for chunked 3R: local alignment plus compressed global state |
| [Point Cloud Mamba](https://arxiv.org/abs/2403.00762) | 2024 | architecture_transfer | Serializes point clouds while preserving neighborhood adjacency | paper | External map memory can be scanned as spatial sequence |
| [MambaVision](https://arxiv.org/abs/2407.08083) / [GitHub](https://github.com/NVlabs/MambaVision) | 2024/2025 | architecture_transfer | Hybrid Mamba-Transformer vision backbone | paper, code | Hybrid design suggests not replacing all attention blindly |
| [MambaOut](https://arxiv.org/abs/2405.07992) / [GitHub](https://github.com/yuweihao/MambaOut) | 2024/2025 | negative_control | Removes SSM token mixer and shows SSM is not always needed for vision classification | paper, code | Forces Dream to justify SSM only for long-sequence / streaming / autoregressive geometry |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | 2025 | architecture_transfer | Kimi Delta Attention; fine-grained finite-state memory gating; hybrid full/linear attention | paper, code/checkpoints claimed | Alternative to Mamba for long-sequence 3R state update |
| [RAM-Net](https://arxiv.org/abs/2602.11958) | 2026 | architecture_transfer | Selectively addressable memory for linear attention | paper | Conceptual support for "fixed state + random-access sparse map" |
| [Adaptive Graph of Thoughts](https://arxiv.org/abs/2502.05078) | 2025 | background | Adaptive test-time reasoning graph | paper | Inspires adaptive geometry hypothesis expansion, but not directly 3R |
| [CTRL](https://arxiv.org/abs/2502.03492) / [GitHub](https://github.com/HKUNLP/critic-rl) | 2025 | architecture_transfer | Reinforcement-trained critic improves iterative critique-revision at test time | paper, code | Blueprint for a geometry critic that spends compute only on inconsistent regions |
| [SEAL](https://arxiv.org/abs/2506.10943) / [GitHub](https://github.com/Continual-Intelligence/SEAL) | 2025 | architecture_transfer | Model generates self-edits / update directives for adaptation | paper, code | Blueprint for controlled adapter updates in 3R under new scene/device distributions |

## Dynamic / 4D / Sensor / Demo Enablers

| Source | Year | Tag | Mechanism | Evidence | Dream relevance |
|---|---:|---|---|---|---|
| [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) | 2024 | demo_enabler | Dynamic scene rendering with 4D representation / deformation | paper, code/project | Strong teacher-facing visualization layer |
| [4D Gaussian Splatting: Towards Efficient Novel View Synthesis](https://huggingface.co/papers/2402.03307) | 2024 | demo_enabler | Anisotropic 4D XYZT Gaussians for dynamic scenes | paper, code | Another 4DGS path; verify repo maturity before use |
| [4DGS in the Wild](https://arxiv.org/abs/2411.08879) | 2024 | demo_enabler | Monocular casual video 4DGS with uncertainty regularization; notes SfM initialization failure | paper | Directly motivates 3R-as-initializer |
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 2023 | demo_enabler | Real-time 3D Gaussian representation and viewer ecosystem | code, paper known | Base viewer/rendering layer for KYKT output demos |
| [InstantSplat](https://github.com/NVlabs/InstantSplat) | 2024 | demo_enabler | Sparse-view SfM-free Gaussian Splatting; supports 3DGS/2DGS/Mip-Splatting path | code, Docker/conda path | Strong visual demo, but heavier dependency stack |
| [Splatt3R](https://github.com/btsmart/splatt3r) | 2024 | demo_enabler | Feed-forward model predicting 3D Gaussians from uncalibrated image pairs | code, Gradio, checkpoint, noncommercial license | Strongest quick visual demo candidate if license is acceptable for class/research |
| [NoPoSplat](https://github.com/cvg/NoPoSplat) | 2025 | demo_enabler | Sparse unposed images to canonical 3D Gaussians, NVS and pose estimation | code, MIT, checkpoints | Good pose-free GS comparator; training is heavy but inference may be viable |
| [GS-CPR](https://github.com/XRIM-Lab/GS-CPR) | 2025 | demo_enabler | Gaussian Splatting camera pose refinement using MASt3R/SfM-style initialization | code | Bridge module for 3R-to-GS refinement |
| [MoSca](https://huggingface.co/papers/2405.17421) | 2024 | demo_enabler | Dynamic Gaussian fusion with motion scaffolds | paper, code | Good comparison for dynamic 4D asset demo |
| [SplineGS](https://huggingface.co/papers/2412.09982) | 2024 | demo_enabler | COLMAP-free monocular dynamic Gaussian reconstruction with splines | paper | Good lower-cost 4D demo candidate if code is available |
| [Hybrid 3D-4DGS](https://github.com/ohsngjun/3D-4DGS) | 2025 | demo_enabler | Static 3D + dynamic 4D Gaussian split | code claimed | Good explanatory dynamic-scene route, but heavier |
| [Instant4D](https://github.com/Zhanpeng1202/Instant4D) | 2025 | demo_enabler | Minute-scale 4D reconstruction stack with external modules | code claimed | Visually attractive but heavy engineering stack |
| [EAG3R](https://arxiv.org/abs/2512.00771) | 2025/2026 | architecture_transfer | Event-augmented pointmap geometry using MonST3R backbone and event/RGB fusion | paper | Confirms Event-DUSt3R idea is already emerging; use as reference not duplicate |
| [Interp3R](https://gist.science/paper/2603.14528) | 2026 | demo_enabler | Continuous-time pointmap interpolation using frames + events | secondary source currently | Needs primary arXiv verification; promising for continuous-time story |
| [Event-3DGS](https://github.com/lanpokn/Event-3DGS) | 2024 | demo_enabler | Event-camera Gaussian Splatting pipeline | code | Research branch; hardware/data/pose preprocessing likely heavy |
| [Next Best Sense](https://github.com/armlabstanford/NextBestSense) | 2025 | demo_enabler | Active sensing / next-best-view system with robot stack | code/docker claimed | Conceptually relevant, but too hardware-heavy for first demo |

## Initial Gaps

1. Need official GitHub/code status verification for 2026 sources: LoGeR, Mem3R, RayMap3R, PAS3R, Interp3R.
2. Need licensing and checkpoint availability for any repo before recommending reproduction.
3. Need KYKT sample compatibility check: image-only, video, dynamic video, long sequence, or special sensor input.
4. Need separate "paper result vs Dream hypothesis" labels for every derived idea.
5. Need direct paper verification for Splatt3R / InstantSplat / MV-DUSt3R+ / NoPoSplat before citing in formal proposal.

## Subagent-Derived Mechanism Vocabulary

Use this vocabulary when converting papers into Dream units:

| Mechanism | 3R meaning | Reject if |
|---|---|---|
| route scan | token/order policy for images, frames, pointmaps, pointers, or voxels | it is only "use Mamba" without route choice |
| persistent state | bounded hidden carrier across frames/chunks | no write/reset/evict policy is specified |
| external spatial memory | explicit growing map indexed by geometry | claimed as O(1) total memory |
| global-local hybrid | local geometry alignment plus sparse/global correction | full attention is still used everywhere |
| critic-revision | detect inconsistency, then locally revise output | it only reruns the same model without a critic |
| self-adapt update | constrained adapter/state update at test time | it is offline fine-tuning renamed as online adaptation |
