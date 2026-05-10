# Dream3R Demo Summary

Status: ready for first-generation research demo.

Date: 2026-05-10

## 30-Second Opening

Dream3R 的初代目标不是做一个孤立的 3R 模型，而是把现有 3R 家族的长处综合成一个可控、可验证、可扩展的重建系统。现在代码已经把 Perceiver、NSA SpatialMemory、Active/Stable AnchorBank、Geometric Critic、Permanence slots、Composer expert routing、Mamba hybrid recurrence 和 GaussianHead contract 接成一个可运行原型。

## Main Claim

Dream3R 已经完成从架构设想到可运行 control-graph 3R prototype 的第一阶段闭环：

- 能 forward。
- 能跨窗口 streaming。
- 能 active state 更新并 promote 到 stable memory。
- 能让 NSA 同时融合 compressed / selected / sliding context。
- 能通过 Critic 产生 conflict 和 repair action。
- 能切换 `cross_attention` 与 `mamba_hybrid` state recurrence。
- 能通过 smoke 和 full unit test suite。

## Live Command

Run on server:

```bash
cd /hdd3/kykt26/code/dream3r
conda run -n dream3r python -m dream3r.demo_mamba_path
```

## Captured Demo Output

Latest captured output:

```json
{
  "device": "cuda",
  "variants": [
    {
      "recurrence_type": "cross_attention",
      "backend": "StateTokenRecurrence",
      "backend_error": "",
      "elapsed_ms_3_windows": 5357.51,
      "latent_state_tokens": [1, 32, 128],
      "state_delta_mean_abs": 0.586081,
      "stable_promotion_rate": 1.0,
      "nsa_branch_mean": {
        "compressed": 0.046747,
        "selected": 0.889811,
        "sliding": 0.063441
      },
      "conflict_score_mean": 74.862854,
      "recommended_action": [1]
    },
    {
      "recurrence_type": "mamba_hybrid",
      "backend": "mamba_ssm",
      "backend_error": "",
      "elapsed_ms_3_windows": 4370.37,
      "latent_state_tokens": [1, 32, 128],
      "state_delta_mean_abs": 0.311841,
      "stable_promotion_rate": 1.0,
      "nsa_branch_mean": {
        "compressed": 0.10086,
        "selected": 0.884492,
        "sliding": 0.014648
      },
      "conflict_score_mean": 13.274281,
      "recommended_action": [5]
    }
  ]
}
```

## How To Explain The Output

- `device: cuda`: demo runs on GPU.
- `recurrence_type`: same Dream3R model can switch state recurrence backend.
- `backend: mamba_ssm`: Mamba path uses the server's real `mamba_ssm` package.
- `latent_state_tokens: [1, 32, 128]`: streaming state shape is stable.
- `state_delta_mean_abs`: state changes across windows, so recurrence is active.
- `stable_promotion_rate: 1.0`: demo threshold makes active-to-stable memory promotion visible.
- `nsa_branch_mean`: NSA is mixing compressed state, selected AnchorBank memory, and sliding recent context.
- `recommended_action`: Critic repair loop is producing an action, not a dead output.

## Important Caveats

- Mamba uses `mamba_ssm.Mamba(use_fast_path=False)`. The installed fast CUDA path has a `causal_conv1d` ABI mismatch, so we use the compatible path.
- Demo inputs are synthetic. This proves architecture integration, not final reconstruction quality.
- GaussianHead is a tensor contract for future 3DGS output. It is not a renderer yet.
- Full 3R quality claims need real data evaluation and ablations.

## 8-Minute Demo Flow

1. **Problem, 60s**: existing 3R methods are strong but fragmented across quality, streaming, memory, verification, and output representation.
2. **Architecture, 90s**: show Perceiver, SpatialMemory, AnchorBank, Critic, Permanence, ComposerRouter, MemoryBus.
3. **Borrowed strengths, 90s**: MASt3R/Spann3R experts, CUT3R recurrence, NSA sparse attention, Mamba state-space trend, 3DGS output direction.
4. **Live demo, 90s**: run `dream3r.demo_mamba_path` and point to backend/state/NSA/promotion/action.
5. **Verification, 60s**: smoke and full tests pass on server.
6. **Next phase, 90s**: real data loader, ablations, Critic calibration, expert routing quality, 3DGS renderer.

## Verification Snapshot

Last verified on server:

- `scripts/sync_verify_server.ps1 -Mode verify`: local/server package files match.
- `scripts/sync_verify_server.ps1 -Mode test`: smoke pass.
- `scripts/sync_verify_server.ps1 -Mode test -FullTests`: smoke plus all `dream3r.tests.test_*` pass.
- `dream3r.demo_mamba_path`: pass.

## Next Phase

Priority after demo:

1. Real sequence data loader and metrics.
2. Ablation: `cross_attention` vs `mamba_hybrid`.
3. Ablation: NSA on/off, active/stable on/off, Critic on/off.
4. Critic calibration on real geometric distributions.
5. Renderer-backed 3DGS only after dependency approval.
