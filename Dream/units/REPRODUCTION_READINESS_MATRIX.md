# Reproduction Readiness Matrix

Last updated: 2026-05-01

Status: first pass with subagent merge. Some license/checkpoint fields still require direct file-level verification before any public/commercial use.

Purpose:

```text
Choose the first Phase 2 smoke test without confusing visual appeal, paper novelty, and engineering risk.
```

## Readiness Labels

- `P0`: first smoke-test candidate
- `P1`: good after one P0 path succeeds
- `P2`: keep as comparator / later integration
- `P3`: research background only for now

## Candidate Matrix

| Candidate | Official source | License / use note | Code / checkpoint / demo | Dependency risk | KYKT fit | Current rank |
|---|---|---|---|---|---|---|
| Splatt3R | [GitHub](https://github.com/btsmart/splatt3r) | CC BY-NC 4.0 | Gradio demo, pretrained model on Hugging Face, example PLY outputs | High; custom CUDA / modified Gaussian rasterization / MASt3R dependency | Strongest pose-free Gaussian visual surprise path, research/demo only | P0 |
| DUSt3R | [GitHub](https://github.com/naver/dust3r) | CC BY-NC-SA 4.0 non-commercial | interactive demo, Docker path, checkpoints | Medium-high; mature, but CUDA/RoPE/Docker choices matter | Foundational baseline and teaching reference | P0 |
| InstantSplat | [GitHub](https://github.com/NVlabs/InstantSplat) | Apache 2.0 per subagent license check; verify file before use | code, Docker option, MASt3R checkpoint dependency, sample scripts | High; submodules, 3DGS dependencies, CUDA kernels, data prep | Strong asset lane, heavier than Splatt3R | P0/P1 |
| MASt3R | [GitHub](https://github.com/naver/mast3r) | CC BY-NC-SA 4.0, dataset/checkpoint notices apply | HF/local demo, checkpoint downloads, MASt3R-SfM path | High; ASMK/retrieval optional complexity and DUSt3R inheritance | Excellent static baseline and matching/composer path | P1 |
| MV-DUSt3R+ | [GitHub](https://github.com/facebookresearch/mvdust3r) | License file present; verify exact terms before use | Gradio demo, all checkpoints, data/eval code released | Medium-high; authors only tested Linux + CUDA 12.4 | Strong multi-view pose-free geometry demo | P1 |
| Fast3R | [GitHub](https://github.com/facebookresearch/fast3r) | FAIR NC Research License | Gradio demo, HF auto-download weights, example inference API | Medium-high; CUDA 12.4 path, network-dependent HF download, do not install cuROPE | Excellent for Composer and high-image-count baseline runner | P1 |
| MonST3R | [GitHub](https://github.com/Junyi42/monst3r) | License file present; verify exact terms | demo.py, demo_data, Gradio/demo notes, real-time/window-wise modes | Medium; dynamic video path can be sensitive | Best dynamic 3R demo and comparator | P1 |
| NoPoSplat | [GitHub](https://github.com/cvg/NoPoSplat) | MIT | pretrained checkpoints on Hugging Face, eval scripts | Medium-high for training; inference/eval easier than training | Good open-license comparator, less immediate Gradio | P2 |
| CUT3R | [GitHub](https://github.com/CUT3R/CUT3R) | License file present; verify exact terms | Google Drive checkpoints, demo.py, demo_ga.py, examples | Medium-high; CUDA 12.1, RoPE compile, viser, linear memory note | Important architecture baseline | P2 |
| Point3R | [GitHub](https://github.com/YkiWu/Point3R) | License file present; verify exact terms | training/finetuning/eval code, Google Drive checkpoint for finetuning | Medium-high; training commands use 8 processes, no simple Gradio found | Explicit spatial memory comparator | P2 |
| STream3R | [GitHub](https://github.com/NIRVANALAN/STream3R) | NTU S-Lab License 1.0 per subagent; verify file before use | README shows inference code, app.py, HF auto-download, StreamSession; subagent flagged release-surface uncertainty | Medium-high; fresh repo and CUDA-dependent, but API is clear | Causal streaming comparator and KV-cache baseline | P2 |
| TTT3R | [GitHub](https://github.com/Inception3D/TTT3R) | License file present; verify exact terms | demo.py, examples, depends on CUT3R checkpoint | Medium-high; CUT3R/RoPE/evo/open3d dependencies | Test-time update comparator | P2 |
| RayMap3R | [Project](https://raymap3r.github.io/) | Code link claimed; verify repo maturity | project page and code link claimed | Unknown | Dynamic/static inference-time branch | P2/P3 |
| LoGeR | [Project](https://loger-project.github.io/) | GitHub link indicated on source pages; verify repo maturity | project/paper; code availability needs direct check | Unknown | Long-context hybrid memory evidence | P3 until code smoke path is clear |
| Mem3R | [Project](https://lck666666.github.io/Mem3R/) | Verify exact terms | project/paper; code state needs direct check | Unknown | Hybrid memory + TTT evidence | P3 until code smoke path is clear |

## Current Phase 2 Smoke-Test Recommendation

Use a two-lane smoke-test strategy:

```text
Lane A: stable geometry baseline
  DUSt3R first, then MASt3R or Fast3R

Lane B: visual surprise
  Splatt3R first, InstantSplat second, MV-DUSt3R+ third
```

Reasoning:

- DUSt3R is the safest foundational demo and gives the teacher a clean reference point for the 3R jump.
- Splatt3R is the fastest route to a visually impressive pose-free Gaussian artifact, but its non-commercial license means it should be framed as research/demo only.
- InstantSplat appears permissively licensed and visually strong, but has more moving parts than Splatt3R.
- MASt3R and Fast3R should anchor the Composer lane after one baseline works.
- MV-DUSt3R+ is strong but more environment-constrained because its README notes Linux/CUDA 12.4 testing.

## Do Before Any Local Clone

1. Confirm exact license if the result may be shown beyond classroom/research.
2. Confirm GPU/CUDA availability and whether Docker is acceptable.
3. Prefer one clean smoke test over installing several heavy repos at once.
4. Record command, input, output artifact, runtime, and failure mode in this file after each attempt.

## Evidence Notes

- Splatt3R README states it has a Gradio demo, Hugging Face pretrained model, downloadable example images/PLY files, and CC BY-NC 4.0 license.
- MV-DUSt3R+ README states all checkpoints and a Gradio demo were released, and notes Linux/CUDA 12.4 testing.
- Fast3R README states `python fast3r/viz/demo.py` auto-downloads pretrained weights from Hugging Face and provides a Gradio interface.
- DUSt3R and MASt3R both provide interactive demo paths and Docker instructions.
- NoPoSplat provides MIT license and Hugging Face checkpoints, but no first-pass Gradio path was found.
- CUT3R provides Google Drive checkpoints and demo scripts, but notes linear memory growth in its current parallel encoder acceleration path.
- Point3R provides training/finetuning/evaluation code and an explicit spatial pointer memory design; first-pass local demo path is less direct than Fast3R/Splatt3R.
- STream3R provides released inference code/weights and `StreamSession` for sequential input, making it the clearest causal-streaming comparator after the first visual demo.
- TTT3R provides demo commands around CUT3R checkpoints and is a good test-time update comparator, but it inherits CUT3R setup cost.

## Subagent Merge Notes

The reproduction-readiness subagent ranked:

```text
Splatt3R -> DUSt3R -> InstantSplat -> MASt3R -> MV-DUSt3R+ -> Fast3R -> CUT3R
```

This matrix adopts that ordering for first smoke-test planning, with two corrections:

- STream3R is not treated as first-wave ready, but its README does show HF auto-download inference and `StreamSession`; keep it as a causal-streaming comparator, not as a teacher-demo target.
- TTT3R has an official GitHub repo with CUT3R-based demo commands, but it remains an add-on path because it inherits CUT3R setup and checkpoint cost.
