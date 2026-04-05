# KYKT Project Sync

Last updated: 2026-04-06 (v0.4.0 — dark theme + multi-image + viewer)

## 1. Project Overview

This workspace currently contains several related but distinct 3D vision lines:

1. `MVSNet` on `DTU`
   - task: supervised multi-view stereo reconstruction
   - outputs: depth maps, fused point clouds, `.ply`

2. `SfMLearner` on `KITTI`
   - task: self-supervised monocular depth estimation
   - outputs: KITTI depth metrics and checkpoints

3. `DUSt3R`
   - task: pointmap-based 3D reconstruction from image pairs or image sets
   - outputs: match visualizations, point clouds, poses, focals

4. `MonST3R`
   - not integrated yet
   - planned next-stage model after DUSt3R
   - target use: video and dynamic-scene reconstruction


## 2. Workspace Layout

Root:

- `E:\kykt`

Important subfolders:

- `E:\kykt\Coding\3.16`
  - MVSNet work
- `E:\kykt\Coding\3.23`
  - SfMLearner work
- `E:\kykt\Coding\3.30`
  - DUSt3R work
- `E:\kykt\Coding\4.06\vision_ui`
  - local offline frontend MVP

Important docs:

- `E:\kykt\README.md`
- `E:\kykt\KYKT.md`
- existing Chinese-named progress notes under the root and `Coding\3.23`


## 3. Proven Results So Far

### 3.1 MVSNet

Local result directory:

- `E:\kykt\Coding\3.16\d192_fast16_eval`

Status:

- training, inference, and fusion have already been run
- fused point cloud outputs already exist

Server references used in prior work:

- `/hdd3/kykt26/code/MVSNet/checkpoints/d192_fast16`
- `/hdd3/kykt26/code/MVSNet/outputs/d192_fast16_eval`


### 3.2 SfMLearner

Local repo:

- `E:\kykt\Coding\3.23\SfmLearner-Pytorch-master`

Server repo:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master`

KITTI data on server:

- `/hdd3/kykt26/data/kitti/rectified`

Main current result:

- `sequence_length = 5`
- `smooth_loss = 2.0`
- `batch_size = 8`

Formal KITTI Eigen split result:

- `abs_diff = 3.7107`
- `abs_rel = 0.2095`
- `sq_rel = 1.5885`
- `rms = 6.7106`
- `log_rms = 0.2820`
- `abs_log = 0.2067`
- `a1 = 0.6767`
- `a2 = 0.8862`
- `a3 = 0.9573`

This remains the main self-supervised depth baseline.

Key checkpoint directory:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master/checkpoints/rectified,10epochs,epoch_size1000,seq5,b8,s2.0/03-29-10:55`


### 3.3 DUSt3R

Local repo:

- `E:\kykt\Coding\3.30\dust3r-main`

Server repo:

- `/hdd3/kykt26/code/dust3r-main`

Current state:

- server-side DUSt3R pair testing has been run
- match visualization and point cloud export were demonstrated
- local examples proved that the repo, weights, and scripts can produce usable outputs

Current server conventions:

- test images:
  - `/hdd3/kykt26/code/dust3r-main/test_images`
- outputs:
  - `/hdd3/kykt26/code/dust3r-main/outputs`
- model weight:
  - `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`

Important clarification:

- DUSt3R supports both pairs and image sets
- current remote runner v1 still uses the first two uploaded images
- the frontend should not split pair vs multi-image too early
- the backend runner is what needs to evolve toward true multi-image support


## 4. Local Frontend Project

Active project:

- `E:\kykt\Coding\4.06\vision_ui`

Purpose:

Build a local offline frontend that:

1. runs on the local PC
2. communicates with the server only through `ssh / scp`
3. uploads images or video
4. triggers remote model jobs
5. downloads results back into a local cache
6. shows progress, logs, outputs, and downloads in one dashboard

Current project files:

- `app.py`
- `job_store.py`
- `ssh_runner.py`
- `templates/index.html`
- `templates/job_detail.html`
- `static/style.css`
- `static/index.js`
- `static/job_detail.js`
- `static/ply_viewer.js` — Three.js point cloud viewer
- `runners/dust3r_runner.py` — server-side multi-image DUSt3R runner
- `requirements.txt`

Current local cache structure:

```text
local_jobs/
  <job_id>/
    input/
    output/
    logs/
    job.json
    status.json
```

What the current version supports:

- local job creation from the browser
- local input caching
- automatic filename normalization for uploads
  - arbitrary upload names are accepted
  - files are stored internally as stable names like `input_01.ext`
- `job.json` and `status.json` generation
- input preview rendering in the browser
- **drag-and-drop file upload** with file chips and remove buttons
- remote dispatch through system `ssh / scp`
- **multi-image DUSt3R support** (N images, not just pairs)
  - uses `GlobalAlignerMode.PointCloudOptimizer` for N>2
  - unified `dust3r_runner.py` uploaded to server automatically
- result download back into the local cache
  - now also downloads `scene_meta.json` (poses, focals)
- **browser point cloud viewer** (Three.js + OrbitControls)
  - inline PLY rendering with orbit, zoom, pan
  - auto-centers and auto-scales the point cloud
  - reset view button + download link
- live progress polling on:
  - home page (3s interval)
  - job detail page (2.5s interval)
- approximate task progress bars with gradient styling
- **elapsed time counter** on job detail page
- live local log cache rendering with auto-scroll
- task actions:
  - run
  - retry current job
  - duplicate as a new job
- **premium dark theme** with:
  - glassmorphism cards
  - gradient accents (cyan → indigo)
  - Inter + JetBrains Mono fonts
  - smooth micro-animations
  - status glow effects
  - responsive layout

What is still missing:

- MonST3R remote runner
- cancel action for a running remote process
- richer parameter controls in the UI
- reusable presets


## 5. SSH / SCP Setup

Chosen transport for MVP:

- system `ssh`
- system `scp`

Reason:

- no extra inbound service is required on the server
- current environment already supports SSH
- Windows local machine can reuse OpenSSH directly
- enough for upload, trigger, download, and log sync

Current alias in local SSH config:

- `KYKT-UI`

Server target:

- host: `172.17.140.97`
- user: `kykt26`
- remote root: `/hdd3/kykt26`

The current frontend assumes that:

- DUSt3R repo exists at `/hdd3/kykt26/code/dust3r-main`
- DUSt3R weight exists at `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- remote conda env name is `dust3r`


## 6. Current Remote Job Flow

Current DUSt3R remote flow (v2 — multi-image):

1. create local job
2. normalize uploaded filenames locally
3. create remote directories:
   - `/hdd3/kykt26/jobs/<job_id>/input`
   - `/hdd3/kykt26/jobs/<job_id>/output`
   - `/hdd3/kykt26/jobs/<job_id>/logs`
4. upload all normalized inputs (with per-file progress)
5. upload `job.json`
6. upload `dust3r_runner.py` to `/hdd3/kykt26/runners/`
7. run unified runner via SSH with conda env
8. stream remote stdout back into local `.live.log` files
9. auto-detect phase transitions from log content
10. download:
    - `matches.png`
    - `pointcloud.ply`
    - `scene_meta.json` (poses, focals, point count)
    - remote logs


## 7. Immediate Next Goals

Priority order:

1. ~~improve remote observability further~~ ✅ done
   - elapsed timer, auto-scroll logs, per-file upload progress

2. ~~extend DUSt3R from pair-only to real image-set support~~ ✅ done
   - unified `dust3r_runner.py` handles 2..N images
   - uses `PointCloudOptimizer` for N>2

3. ~~add browser-side point cloud preview~~ ✅ done
   - Three.js inline viewer with orbit controls

4. integrate MonST3R as the next model runner
   - keep the same job architecture
   - add a `monst3r_runner.py` on the server side
   - support video or frame-sequence inputs

5. add stronger task controls
   - cancel running remote processes
   - rerun with modified parameters
   - reusable presets

6. validate multi-image DUSt3R end-to-end on the server
   - test with 3+ images
   - verify scene_meta.json output
   - tune point size and viewer behavior


## 8. Guidance for Future Agents

If another agent picks this up, the current best continuation path is:

1. read this file first
2. keep the frontend architecture centered on `job.json` + local cache + SSH/SCP
3. do not split DUSt3R pair and multi-image into separate frontend products
4. expand the backend runners instead
5. preserve the current server assumptions unless the user changes them explicitly

Current most important active repo:

- `E:\kykt\Coding\4.06\vision_ui`
