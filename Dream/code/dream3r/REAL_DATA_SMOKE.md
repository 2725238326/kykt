# Dream3R Real Data Smoke

Status: W19 first slice in progress.

Date: 2026-05-11

## Goal

Move beyond synthetic-only evidence by running the Dream3R control graph on a real RGB/depth sequence. This is a smoke/evidence path, not a trained-quality benchmark.

## First Dataset

KITTI rectified data is used first because the server already has paired files:

- RGB: `/hdd3/kykt26/data/kitti/rectified/*/*.jpg`
- Depth: `/hdd3/kykt26/data/kitti/rectified/*/*.npy`

The loader converts each depth map into a sampled pointmap with approximate scaled KITTI intrinsics, and creates deterministic RGB/depth patch features for the existing no-backbone model path.

## Command

```bash
conda run -n dream3r python -m dream3r.evaluate_real_sequence \
  --data-root /hdd3/kykt26/data \
  --max-sequences 1 \
  --max-windows 2 \
  --recurrence mamba_hybrid \
  --output demo_artifacts/real_sequence/kitti_metrics.json
```

## Expected Output

The JSON contains:

- real-data metric summary from `Evaluator`
- per-window latency
- AnchorBank occupancy
- NSA branch means
- stable promotion rate
- selected anchor 3D distance
- Critic conflict score
- recommended repair action

## Interpretation Boundary

This path proves that real RGB/depth windows now flow through Dream3R's Perceiver-free feature path, SpatialMemory, AnchorBank, Permanence, Composer, and Critic. It does not claim trained accuracy until weights and real-data training/evaluation are added.
