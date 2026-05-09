# Dream3R v0.3 Code Review Prompt

Use this prompt to onboard a new agent (Claude, Codex, or human) for reviewing,
extending, or debugging the Dream3R v0.3 codebase.

---

## Context

Dream3R is a streaming 3D reconstruction architecture organized around 6 modules
communicating through a typed memory bus (C6). The project lives at:

- **Local (Windows):** `E:\kykt\Dream\code\dream3r\` — editing, orchestration, markdown
- **Server (Linux):** `/hdd3/kykt26/code/dream3r/dream3r/` — GPU execution, training

SSH: `ssh BUAA-Server` (config alias for `kykt26@172.17.140.97`)

## Architecture at a glance

```
C1 Perceiver ─── T1 frame_tokens, T2 pointmap, T3 evidence (17 signals)
      │
C3 Permanence ── dynamic_ratio, suppress_static_write ──┐
      │                                                   │ CR-2
C2 SpatialMemory ── NSA(compressed+selected+sliding) ────┤
      │              AnchorBank(K=256, bus-gated writes)  │ CR-3
      │              StateTokenRecurrence(32 tokens)      │
      │                                                   │
C5 ComposerRouter ── 7 experts, cost-normalized routing ──┤
      │                                                   │ CR-1
C4 Critic ── conflict_score, repair_action ───────────────┘
      │
C6 MemoryBus ── publish/read/handoff + CR-1..CR-6 gates
```

## File map (29 Python files)

### Core modules
| File | What it does | Key classes |
|------|-------------|-------------|
| `model.py` | Top-level orchestrator, one bus tick = one window | `Dream3R`, `build_dream3r` |
| `modules.py` | C1-C5 computational cores (v01 + v03) | `Perceiver`, `SpatialMemory`, `Permanence`, `Critic`, `ComposerRouter` |
| `bus.py` | C6 typed tensor namespace, CR-1..CR-6 | `MemoryBus`, `BusSignal`, `EvidenceLabel` |
| `nsa_attention.py` | 3-branch Native Sparse Attention | `NSAAttention`, `CompressedBranch`, `SelectedBranch`, `SlidingBranch` |
| `anchor_bank.py` | Bounded spatial K/V memory | `AnchorBank`, `WriteResult`, `ReadResult` |
| `losses.py` | Multi-task loss (7 base + 3 v0.3 terms) | `Dream3RLoss` |
| `config.py` | YAML config + presets | `load_config`, `PRESETS`, `config_to_model_args` |

### Composer experts (7 adapters)
| File | Expert | Latency | Strength |
|------|--------|---------|----------|
| `composer_experts/mast3r_adapter.py` | MASt3R | 35ms | indoor_static 0.9 |
| `composer_experts/fast3r_adapter.py` | Fast3R | 12ms | dense_sequential 0.8 |
| `composer_experts/spann3r_adapter.py` | Spann3R | 28ms | dense_sequential 0.95 |
| `composer_experts/cut3r_adapter.py` | CUT3R | 30ms | state token recurrence |
| `composer_experts/moge2_adapter.py` | MoGe-2 | 18ms | sparse_view 0.9 |
| `composer_experts/depthanything_adapter.py` | DAv2 | 8ms | monocular depth |
| `composer_experts/test3r_adapter.py` | Test3R | 120ms | offline verification |

All adapters are **stubs** (random projections). Real model loading is TODO.

### Training infrastructure
| File | Purpose |
|------|---------|
| `train.py` | DDP, AMP, multi-stage LR with freeze/unfreeze, checkpoint I/O |
| `data/synthetic.py` | Deterministic synthetic sequences + DTU stub |
| `evaluate.py` | Evaluator (pointmap MSE, critic F1, branch usage, routing entropy) |
| `bench_frame_budget.py` | Per-module p50/p95/p99 latency profiler |

### Tests
| File | Covers |
|------|--------|
| `smoke_test.py` | 9-section integration test (v03 forward/backward/bus/NSA/AnchorBank/experts/v01-compat/dataset) |
| `tests/test_nsa_attention.py` | NSA branch shapes, masks, gradients |
| `tests/test_anchor_bank.py` | Write/read/gating/quarantine/prune/batch |
| `tests/test_composer_experts.py` | Registry, capability matrix, adapter forward |
| `tests/test_spatial_memory.py` | SpatialMemory + ComposerRouter init/forward/multi-window/gradient |

## Key contracts to preserve

1. **Bus ownership**: every signal has a single owner in `bus.py:_owner_table`. Adding new signals requires registering ownership here.
2. **CR gates**: CR-1 (reroute spread gate), CR-2 (permanence write suppress), CR-3 (retrieval depth + confidence/permanence bias), CR-4 (tiebreak), CR-5 (label propagation), CR-6 (audit log).
3. **AMP safety**: all mask fills must use `torch.finfo(dtype).min` not `-1e9`. All bank writes must `.float()` before storing.
4. **v01/v03 dual mode**: `model.py` supports `version="v01"` and `version="v03"` via config. v01 classes have `_v01` suffix and are preserved for ablation.
5. **Tensor shapes** (v03 small preset):
   - Input: `[B, N=4, P=196, D=768]`
   - State tokens: `[B, S=32, D_mem=128]`
   - AnchorBank: capacity 256, key/value `[B, 256, 128]`
   - Evidence: `[B, N, 17, 32]`

## Known gaps (not yet implemented)

| ID | Gap | Priority |
|----|-----|----------|
| A4 | AnchorBank lacks `points3d` payload (spec requires storing 3D point anchors) | Medium |
| A5 | DINOv3-S backbone not integrated (Perceiver still uses ViT-Base/identity) | Medium |
| A6 | Test3R lazy invocation path (Critic triggers off-path verification) | Medium |
| C1 | DTUDataset is a stub (returns random tensors) | Medium |
| C2 | No data augmentation | Low |
| D1-D4 | No standard depth metrics, pose eval, ECE, or visualization | Medium |
| E1 | No sequence-level streaming orchestration | Medium |
| E2 | All 7 expert adapters are stubs | High |
| E4 | AnchorBank.write still has per-batch Python loop | Low |

## How to verify changes

```bash
# SSH to server
ssh BUAA-Server

# Navigate and activate
cd /hdd3/kykt26/code/dream3r
source activate dream3r

# Clear caches
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Run smoke test (must pass all 9 sections)
CUDA_VISIBLE_DEVICES=0 python -m dream3r.smoke_test

# Run unit tests
python -m dream3r.tests.test_nsa_attention
python -m dream3r.tests.test_anchor_bank
python -m dream3r.tests.test_composer_experts
python -m dream3r.tests.test_spatial_memory

# Profile latency (p95 must be < 50ms)
CUDA_VISIBLE_DEVICES=0 python -m dream3r.bench_frame_budget --preset small --n-windows 30

# Quick training test (should see loss decreasing)
CUDA_VISIBLE_DEVICES=0 python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES']='0'
from dream3r.config import load_config
from dream3r.train import train
cfg = load_config(preset='small', overrides={'gpus':'0','dataset':'synthetic','epochs':5,'batch_size':4,'num_workers':0,'n_train_sequences':50,'n_val_sequences':10,'log_every':5,'eval_every_epoch':5,'save_every_epoch':10})
train(cfg)
"
```

## Review checklist for any PR touching this code

- [ ] `python -m dream3r.smoke_test` passes all 9 sections
- [ ] All 4 unit test files pass
- [ ] `bench_frame_budget.py` p95 < 50ms
- [ ] No new `-1e9` literals (use `torch.finfo(dtype).min`)
- [ ] Any new bus signal registered in `bus.py:_owner_table`
- [ ] Any new module output key added to `model.py` return dict
- [ ] `config.py` updated if new hyperparameters added
- [ ] v01 backward compatibility preserved (don't break `build_dream3r("small_v01")`)
- [ ] No server-only imports at module level (all torch imports are fine; avoid importing CUDA-only packages at top level)

## Spec references (do not re-read full files, cite by section)

- Architecture v0.2: `specs/SPEC-20260506-004-dream3r-architecture-v02.md`
- Ablation plan v0.2: `specs/SPEC-20260506-005-dream3r-ablation-plan-v02.md`
- Memory v0.3 addendum: `specs/SPEC-20260508-001-dream3r-c2-memory-v03-addendum.md`
- Memory ablation addendum: `specs/SPEC-20260508-002-dream3r-memory-v03-ablation-addendum.md`
