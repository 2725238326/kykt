# KYKT Project Sync

Last updated: 2026-04-06

## 1. Project Overview

This workspace currently contains several parallel 3D vision lines. They are related, but they are **not the same task** and should not be mixed together:

1. `MVSNet` on `DTU`
   - task: supervised multi-view stereo reconstruction
   - main outputs: depth maps, fused point clouds, `.ply`

2. `SfMLearner` on `KITTI`
   - task: unsupervised / self-supervised monocular depth estimation
   - main outputs: depth evaluation metrics on KITTI

3. `DUSt3R`
   - task: pointmap-based 3D reconstruction from image pairs or image sets
   - main outputs: matches visualization, point clouds, poses / focals

4. `MonST3R`
   - not yet integrated locally
   - planned next-stage model after DUSt3R
   - target use: video / dynamic scene reconstruction


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
  - new local frontend MVP

Important docs:

- `E:\kykt\README.md`
- `E:\kykt\近期工作历程_3.2-3.23.md`
- `E:\kykt\Coding\3.23\KITTI服务器任务安排.md`
- `E:\kykt\KYKT.md`


## 3. Proven Results So Far

### 3.1 MVSNet

Local result directory:

- `E:\kykt\Coding\3.16\d192_fast16_eval`

Status:

- training / inference / fusion pipeline has been run before
- fused point cloud outputs already exist

Server-side references used in prior work:

- `/hdd3/kykt26/code/MVSNet/checkpoints/d192_fast16`
- `/hdd3/kykt26/code/MVSNet/outputs/d192_fast16_eval`


### 3.2 SfMLearner

Local repo:

- `E:\kykt\Coding\3.23\SfmLearner-Pytorch-master`

Server repo:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master`

Formatted KITTI data on server:

- `/hdd3/kykt26/data/kitti/rectified`

Main useful result:

- sequence length `5`
- smooth loss `2.0`
- batch size `8`

Formal depth evaluation result already obtained on KITTI Eigen split:

- `abs_diff = 3.7107`
- `abs_rel = 0.2095`
- `sq_rel = 1.5885`
- `rms = 6.7106`
- `log_rms = 0.2820`
- `abs_log = 0.2067`
- `a1 = 0.6767`
- `a2 = 0.8862`
- `a3 = 0.9573`

This is currently the main unsupervised depth result.

Key checkpoint directory:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master/checkpoints/rectified,10epochs,epoch_size1000,seq5,b8,s2.0/03-29-10:55`

Important engineering notes:

- many old compatibility issues were patched
- `path.py` style `.isfile()` had to be replaced with `.is_file()`
- old NumPy API such as `np.int` also had to be patched in evaluation scripts
- `m=0` is currently the stable choice for explainability mask


### 3.3 DUSt3R

Local repo:

- `E:\kykt\Coding\3.30\dust3r-main`

Server repo:

- `/hdd3/kykt26/code/dust3r-main`

Current state:

- environment on server was made usable after avoiding broken conda torch path
- DUSt3R pairwise testing has been run
- matches visualization and point cloud export were demonstrated

Known server-side conventions used so far:

- test images:
  - `/hdd3/kykt26/code/dust3r-main/test_images`
- outputs:
  - `/hdd3/kykt26/code/dust3r-main/outputs`
- model weight:
  - `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`

Important clarification:

- DUSt3R can process both image pairs and multiple images
- two-image use is just the easiest first test
- later work should support both pair and multi-image jobs without splitting the architecture too early


## 4. New Active Project: Local Frontend

Current active work is now:

- `E:\kykt\Coding\4.06\vision_ui`

This is intended to become a **local offline frontend** that:

1. runs on the local PC
2. only communicates with the server through `ssh / scp`
3. uploads images or video
4. triggers remote model jobs
5. downloads results back to the local UI
6. displays matches, point clouds, and future model outputs

Current MVP skeleton already created:

- `E:\kykt\Coding\4.06\vision_ui\app.py`
- `E:\kykt\Coding\4.06\vision_ui\job_store.py`
- `E:\kykt\Coding\4.06\vision_ui\ssh_runner.py`
- `E:\kykt\Coding\4.06\vision_ui\templates\index.html`
- `E:\kykt\Coding\4.06\vision_ui\templates\job_detail.html`
- `E:\kykt\Coding\4.06\vision_ui\static\style.css`
- `E:\kykt\Coding\4.06\vision_ui\requirements.txt`

What this MVP already does:

- local web page exists
- supports local job creation
- saves uploaded files under a local job directory
- writes `job.json` and `status.json`
- previews uploaded inputs in the browser
- records model choice such as `dust3r` / `monst3r`
- includes a first working `ssh/scp` integration path for DUSt3R pair jobs
- supports a detail-page trigger that dispatches a remote DUSt3R job
- downloads remote outputs back into the local job cache
- includes a first monitoring layer:
  - job status polling API
  - running-phase progress panel
  - local live log cache rendered in the UI
  - index-page job polling and mini progress cards

What it does **not** do yet:

- no generic remote runner yet
- no MonST3R remote execution yet
- no point cloud rendering in browser yet
- no true server-written `status.json` yet
- no richer task control such as cancel / retry / re-run with modified parameters


## 5. Planned Local Frontend Architecture

Recommended architecture:

### Local side

Run a local FastAPI web UI on:

- `http://127.0.0.1:8000`

Responsibilities:

1. user selects model and files
2. local app creates a `job_id`
3. local app stores job inputs under local cache
4. local app uploads files and `job.json` to the server using `scp`
5. local app starts remote inference using `ssh`
6. local app polls remote `status.json`
7. local app downloads outputs back to local cache
8. local app displays output images and offers point cloud download

### Server side

Only run models and runners. Do not open extra service ports.

Suggested remote job layout:

```text
/hdd3/kykt26/jobs/
  <job_id>/
    input/
    output/
    logs/
    job.json
    status.json
```

Suggested remote entry point:

```text
python /hdd3/kykt26/code/vision_runner/run_job.py --job /hdd3/kykt26/jobs/<job_id>/job.json
```


## 6. Why SSH / SCP Was Chosen

The current plan is to use **system `ssh / scp`** first instead of Paramiko.

Reason:

- the environment already allows SSH
- no extra inbound service is needed on the server
- local Windows machine can reuse existing OpenSSH tooling
- simpler MVP for upload, trigger, download

Possible upgrade later:

- move to Paramiko if fine-grained connection control or streaming logs becomes necessary


## 7. Immediate Next Steps

### Step 1

Connect `vision_ui` to the server for DUSt3R pair jobs:

- first version already wired locally
- next focus is end-to-end validation against the server:
  - upload images via `scp`
  - upload generated `job.json`
  - run remote DUSt3R runner through `ssh`
  - download:
    - matches image
    - point cloud
    - logs

Also improve monitoring usability:

- map low-level phases to user-readable progress states
- avoid full-page refresh UX
- make remote errors and logs easier to understand
- reduce Windows encoding issues in SSH/SCP integration
- improve the homepage into a clearer task center

### Step 2

Standardize remote runner format:

- `run_job.py`
- `dust3r_runner.py`
- `monst3r_runner.py`

### Step 3

Add browser-side result rendering:

- image result cards
- job status
- downloadable `.ply`
- later maybe `three.js` point cloud preview

### Step 4

Integrate MonST3R after DUSt3R flow is stable.


## 8. MonST3R Onboarding Plan

Recommended learning / integration sequence:

1. finish DUSt3R pair job pipeline
2. support DUSt3R multi-image jobs
3. learn MonST3R by first running official inference on demo video
4. only then add MonST3R as another backend runner in the same frontend system

Important idea:

- do **not** build a separate frontend for MonST3R
- the frontend should dispatch by model name
- only the server-side runner should differ


## 9. Important Conventions For Future Agents

1. Do not confuse the task families:
   - `MVSNet != SfMLearner != DUSt3R != MonST3R`

2. Prefer keeping all user-facing tooling separate from model repos:
   - model repos stay mostly untouched
   - orchestration lives in `vision_ui` and future server runners

3. Prefer parameterized runners over hard-coded one-off scripts.

4. Prefer writing status and outputs into structured job directories.

5. When giving the user shell commands with placeholders, avoid fake placeholders like `<your_dir>` unless immediately replaced with a real path or variable example.


## 10. Current Progress Snapshot

Done:

- MVSNet line reproduced enough to output point clouds
- SfMLearner line trained and formally evaluated
- DUSt3R pair testing ran successfully
- local frontend MVP directory created
- local frontend can already create and store jobs locally
- local frontend now contains a first remote DUSt3R dispatch pipeline using system `ssh/scp`
- unified project sync document added

In progress:

- validating the local frontend against the real server-side DUSt3R setup
- improving task observability and usability for remote jobs
- moving from a demo-like task page toward a reusable local job dashboard

Not started:

- remote generic job runner
- MonST3R integration into frontend
- browser point cloud visualization
