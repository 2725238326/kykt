# Server Preparation

Last updated: 2026-04-10

This document tracks what still needs to happen on the server side before the
local frontend can be treated as a more deliverable tool.

## 1. DUSt3R Optimization Track

Current assumptions:

- Repo: `/hdd3/kykt26/code/dust3r-main`
- Weights: `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- Env: `dust3r`

### Immediate validation goal

Run one complete multi-image job with `3` to `5` images and confirm:

- `matches.png` is meaningful and not empty.
- `pointcloud.ply` can be inspected in MeshLab.
- `scene_meta.json` is generated and internally consistent.
- the local frontend receives status updates and final artifacts.

### Optimization checklist

1. Compare `scene_graph=complete` vs `scene_graph=swin-5`.
2. Compare `image_size=224` vs `image_size=512` for speed and quality.
3. Tune `max_points` to balance MeshLab usability and fidelity.
4. Decide whether `match_viz_count` should default to `30`, `50`, or `0`.
5. Record one recommended preset for:
   - quick preview
   - normal pair / small multi-view
   - longer sequence

## 2. MonST3R Deployment Track

Current frontend state:

- model entry already exists in the UI.
- local runner skeleton exists at `runners/monst3r_runner.py`.
- remote execution currently returns a preparation message instead of real inference.

Current server state:

- Repo exists at `/hdd3/kykt26/code/monst3r`.
- Repo commit is `574cc77`.
- Submodules `croco` and `viser` are present.
- Conda env `monst3r` exists.
- Torch check passed: `2.5.1+cu121`, CUDA available, 4 GPUs visible.
- Requirements were installed with `torch` and `torchvision` excluded to preserve the working CUDA stack.
- Official checkpoint download is not complete. A residual interrupted `download_ckpt.sh` / `wget models.zip` process was stopped on 2026-04-10.

### What needs to be prepared on the server

1. Clone the MonST3R repo to a stable location, for example:
   - `/hdd3/kykt26/code/monst3r`
2. Create a dedicated conda environment, for example:
   - `monst3r`
3. Install the required dependencies:
   - PyTorch / CUDA-compatible stack
   - xformers if required by the chosen commit
   - any viewer or geometry dependencies needed by the official demo
4. Download and store the required pretrained checkpoints.
5. Run the official demo manually once on the server.
6. Decide the standard remote output contract.

Progress:

- Steps 1 to 3 are done enough for the first demo attempt.
- Step 4 is the active blocker.

Checkpoint sources from `data/download_ckpt.sh`:

- Main MonST3R checkpoint from Google Drive file id `1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E`, saved under `/hdd3/kykt26/code/monst3r/checkpoints`.
- SEA-RAFT `models.zip` from Dropbox URL `https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip`, unzipped under `/hdd3/kykt26/code/monst3r/third_party/RAFT/models`.
- Additional RAFT checkpoint from Google Drive file id `1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu`, saved under `/hdd3/kykt26/code/monst3r/third_party/RAFT/models`.
- SAM2 checkpoint `sam2.1_hiera_large.pt` from Meta's public file host, saved under `/hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints`.

### Recommended MonST3R output contract

The frontend will be easier to support if the remote runner always writes:

- `output/preview.png`
- `output/result_summary.json`
- `output/trajectory.json` or similar metadata
- `output/pointcloud.ply` if available
- `logs/runner.log`
- `status.json`

### First integration milestone

The first working MonST3R runner does **not** need full polish. It only needs:

- input: one uploaded video or one uploaded frame sequence
- remote inference: official demo-equivalent execution
- outputs: at least one preview image + one metadata file + logs
- local UI: display status, logs, preview image, and downloadable outputs

## 3. Shared Reliability Work

These improvements benefit both DUSt3R and MonST3R:

1. More reliable remote cancellation and verification.
2. Better cleanup of stale remote job directories.
3. Stronger server-written `status.json` phases.
4. A consistent per-job `result_summary.json`.
5. Better retry guidance when a job only partially succeeds.
