# Dream3R Initial Research Closure and Demo Plan

Status: active for tonight closure and first public/internal demo.

Date: 2026-05-10

## Research Thesis

Dream3R 初代研究的核心主张不是再做一个单体 3R 模型，而是把现有 3R 方法的长处组织成一个可控、可验证、可扩展的架构：

- DINO/foundation encoder 提供视觉先验。
- 3R expert adapters 提供 MASt3R / Spann3R / Fast3R 等方法的能力入口。
- NSA sparse attention 负责长序列上下文的高效融合。
- Active/Stable state memory 解决 streaming 过程中的短期更新与长期保留。
- AnchorBank 用几何锚点承载 stable memory。
- Critic 用几何一致性信号驱动 repair/reroute。
- Permanence 用 slot 和 ISA-style reference frames 维护对象连续性。
- Mamba hybrid recurrence 为 streaming state evolution 提供下一代时序路径。
- GaussianHead 定义未来 3DGS 输出契约。

初代目标：证明这些模块不是散装概念，而是已经接成一个可运行、可测试、可展示的系统原型。

## Tonight Definition of Done

今晚结束时必须满足：

1. `dream3r.demo_mamba_path` 能在服务器跑通并输出可讲的 JSON。
2. Smoke test 通过。
3. Full `dream3r.tests.test_*` suite 通过。
4. 展示文档能解释：
   - 我们解决什么问题。
   - 从 3R 家族吸收了什么优点。
   - Dream3R 架构如何综合这些优点。
   - 当前代码已经真实做到哪些。
   - 下一阶段还缺什么。
5. 不再临时开新技术方向；今晚只收束、验证、展示。

## Demo Storyline

### 1. Problem

现有 3R 方法各有强项，但很难同时满足：

- 高质量视觉先验
- 长序列 streaming
- bounded memory
- 几何一致性自检
- 多专家路由
- 对象级连续性
- 面向未来 3DGS 输出

Dream3R 的切入点是 control-graph architecture：把重建、记忆、验证、路由、对象保持都变成可交互的模块，而不是一个黑盒 forward。

### 2. What We Borrowed From Existing 3R

- MASt3R / Spann3R：成熟 3R expert 能力与 pointmap-style 几何输出。
- CUT3R：state-token recurrence 的 streaming 思路。
- Point3R / Mem3R / LONG3R 方向：长期空间记忆与遗忘问题意识。
- VGGT / foundation model trend：统一视觉先验的重要性。
- NSA sparse attention：compressed / selected / sliding 三路上下文融合。
- Mamba trend：用 state-space path 改进长序列时序演化。

### 3. What Dream3R Adds

- MemoryBus：模块间 typed handoff 和 CR gates。
- AnchorBank：stable spatial memory，而不是无边界 token cache。
- Active/Stable state split：active state 负责窗口内演化，stable state 负责长期可召回。
- Geometric Critic：Sampson/depth/covisibility 信号进入 conflict 和 repair loop。
- Permanence + ISA slots：对象 slot 带 reference frame，能跨窗口匹配。
- ComposerRouter：按能力和成本路由到 expert。
- MambaHybridRecurrence：`state_recurrence_type="mamba_hybrid"` 已接入可运行。
- GaussianHead contract：为 3DGS 输出保留明确 tensor schema。

## Live Demo Commands

Run on server:

```bash
cd /hdd3/kykt26/code/dream3r
conda run -n dream3r python -m dream3r.demo_mamba_path
```

Expected talking points from JSON:

- `device: cuda`
- `cross_attention` and `mamba_hybrid` both run for 3 streaming windows.
- `mamba_hybrid.backend` should be `mamba_ssm`.
- `latent_state_tokens` remains `[1, 32, 128]`.
- `state_delta_mean_abs` shows state is evolving.
- `stable_promotion_rate` shows active state can promote into stable memory.
- `nsa_branch_mean` shows compressed / selected / sliding branches are active.
- `recommended_action` shows Critic repair loop is producing downstream actions.

Validation:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File E:\kykt\Dream\code\dream3r\scripts\sync_verify_server.ps1 -Mode test
powershell -NoProfile -ExecutionPolicy Bypass -File E:\kykt\Dream\code\dream3r\scripts\sync_verify_server.ps1 -Mode test -FullTests
```

## Tonight Execution Plan

### Phase A: Freeze Scope

Do:

- Keep W1-W18 as the initial research boundary.
- Treat W17 Mamba demo path as the headline new result.
- Treat W18 GaussianHead as future-output contract, not renderer completion.

Do not:

- Start new renderer work.
- Install packages.
- Download checkpoints.
- Rewrite core modules unless a test fails.

### Phase B: Evidence Pack

Prepare these artifacts:

- `CYCLE_033_PLAN.md`: full architecture advancement record.
- `CYCLE_034_PLAN.md`: stabilization and Mamba path status.
- `DEMO_2026_05_11.md`: short demo brief.
- `INITIAL_RESEARCH_DEMO_PLAN.md`: this closure plan.
- Test output summary: smoke + full suite pass.
- Demo output JSON.

### Phase C: Demo Flow

Suggested 8-minute version:

1. 60s: Current 3R gap and Dream3R thesis.
2. 90s: Architecture graph: Perceiver, SpatialMemory, Permanence, Critic, Composer, Bus.
3. 90s: What was integrated from 3R family and why it matters.
4. 90s: Mamba path live run.
5. 60s: Tests and verification.
6. 90s: Next stage: real data, calibration, renderer/3DGS, ablation.

### Phase D: Final Stop Criteria

Stop tonight when:

- Demo command output is captured.
- Full tests pass after latest sync.
- Docs are synced to server.
- The remaining work is clearly labeled as next phase, not unfinished current phase.

## Next Phase After Demo

Priority order:

1. Real-data loader and real sequence evaluation.
2. Ablation table: cross-attention vs Mamba hybrid, with NSA on/off and active/stable on/off.
3. Critic calibration on real geometric consistency distributions.
4. Fast3R dependency cleanup if approved.
5. 3DGS renderer backend only after `gsplat` or equivalent is approved.

## One-Sentence Summary

Dream3R 初代研究已经完成从“架构设想”到“可运行控制图 3R 原型”的转变；今晚的任务是冻结范围、跑通展示、拿测试结果支撑这个结论。
