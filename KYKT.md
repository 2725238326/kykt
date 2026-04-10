# KYKT Project Sync

Last updated: 2026-04-11

## Project Lines

This workspace currently has four related but separate 3D vision lines:

- `MVSNet` on `DTU`: supervised multi-view stereo reconstruction. Main outputs are depth maps and fused `.ply` point clouds.
- `SfMLearner` on `KITTI`: self-supervised monocular depth. Main output is KITTI depth metrics and checkpoints.
- `DUSt3R`: pointmap-based 3D reconstruction from image pairs or image sets. Main outputs are match visualizations, point clouds, poses, and focals.
- `MonST3R`: next-stage dynamic/video reconstruction model. Server repo, env, Python dependencies, and required checkpoints are now in place; the local client is being wired to run official demo-equivalent inference and pull back artifacts.

## Workspace Layout

- `E:\kykt`: project root.
- `E:\kykt\Coding\3.16`: MVSNet work.
- `E:\kykt\Coding\3.23`: SfMLearner work.
- `E:\kykt\Coding\3.30`: DUSt3R work.
- `E:\kykt\Coding\4.06\vision_ui`: current local frontend project.
- `E:\kykt\Coding\external_sources`: local copies of external upstream repos kept for reference or upload preparation.

## Proven Results

### MVSNet

Known result directory:

- `E:\kykt\Coding\3.16\d192_fast16_eval`

Prior server references:

- `/hdd3/kykt26/code/MVSNet/checkpoints/d192_fast16`
- `/hdd3/kykt26/code/MVSNet/outputs/d192_fast16_eval`

### SfMLearner

Server repo:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master`

KITTI data:

- `/hdd3/kykt26/data/kitti/rectified`

Main useful result:

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

Key checkpoint:

- `/hdd3/kykt26/code/SfmLearner-Pytorch-master/checkpoints/rectified,10epochs,epoch_size1000,seq5,b8,s2.0/03-29-10:55`

### DUSt3R

Local repo:

- `E:\kykt\Coding\3.30\dust3r-main`

Server DUSt3R repo:

- `/hdd3/kykt26/code/dust3r-main`

Server model:

- `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`

Server MonST3R repo:

- `/hdd3/kykt26/code/monst3r`

Local MonST3R source copy:

- `E:\kykt\Coding\external_sources\monst3r_official_20260410`

Server MonST3R env:

- `monst3r`

Current MonST3R status:

- Repo was uploaded from local clone because direct server GitHub clone was unreliable.
- Commit is `574cc77`.
- Submodules `croco` and `viser` are present.
- Conda env `monst3r` was cloned from the working `dust3r` env.
- Torch check passed in that env: `2.5.1+cu121`, CUDA available, 4 GPUs visible.
- Python requirements were installed after excluding `torch` and `torchvision` to avoid breaking the CUDA stack.
- Required checkpoints are now present on the server:
  - `/hdd3/kykt26/code/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`
  - `/hdd3/kykt26/code/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth`
  - `/hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt`
- A residual interrupted `download_ckpt.sh` / `wget models.zip` process was found and stopped on 2026-04-10.
- The temporary local upload area `E:\kykt\Coding\4.10` was cleaned up on 2026-04-10.
- The local MonST3R tarball was deleted after successful server extraction.
- Server upload archives were moved out of `/hdd3/kykt26/code` into `/hdd3/kykt26/archive/code_uploads_20260410`.
- Local helper script for uploading manually downloaded MonST3R weights: `E:\kykt\tools\upload_monst3r_weights.ps1`.
- Local helper script for checking remote MonST3R readiness through pure SSH: `E:\kykt\tools\check_monst3r_remote.ps1`.
- Local helper script for launching the official MonST3R demo on the server after weights are ready: `E:\kykt\tools\run_monst3r_demo_remote.ps1`.
- Local weight staging directory, ignored by git: `E:\kykt\model_uploads\monst3r`.
- Current recommended server workflow is terminal-first: use Electerm or another SSH/SFTP client for remote browsing, and use the helper scripts above for validation / launch. VS Code Remote-SSH is currently optional, not required.

Notes:

- DUSt3R supports both image pairs and image sets.
- The frontend should not split pair and multi-image into separate products.
- Backend runners should handle whether a job has 2 images or N images.

## Local Frontend

Active project:

- `E:\kykt\Coding\4.06\vision_ui`
- `E:\kykt\Coding\4.06\vision_ui\client` (new React + TypeScript rebuild skeleton)

Purpose:

- Run a local browser UI.
- Communicate with the server only through system `ssh` and `scp`.
- Upload images, videos, or frame sequences.
- Trigger remote model jobs.
- Download results into local cache.
- Display progress, logs, images, and point clouds locally.

Current local cache layout:

```text
local_jobs/
  <job_id>/
    input/
    output/
    logs/
    job.json
    status.json
```

Current frontend capabilities:

- Browser-based local job creation.
- A new Apple-inspired client rebuild skeleton now exists under `E:\kykt\Coding\4.06\vision_ui\client`.
- The rebuild stack choice is React + TypeScript first, with Tauri recommended as the future desktop shell.
- 2026-04-10: The React rebuild now has a richer task detail view with result summary cards, output action cards, and local output opening through the FastAPI bridge.
- 2026-04-10: A Tauri 2 desktop shell scaffold now exists under `E:\kykt\Coding\4.06\vision_ui\client\src-tauri`.
- 2026-04-10: The Tauri desktop release now supervises the local FastAPI backend automatically. It starts `.venv\Scripts\python.exe -m uvicorn app:app` from the project root and writes backend logs to `E:\kykt\Coding\4.06\vision_ui\local_jobs\_desktop\backend.log`.
- 2026-04-11: The desktop/rebuild client backend was moved to the dedicated local port `127.0.0.1:8765` to avoid accidentally reusing old `8000` debug backends.
- 2026-04-11: Easy-to-find desktop release copies are available at `E:\kykt\release\kykt_vision_client`.
- Desktop build artifacts are also available at:
  - `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\kykt_vision_client.exe`
  - `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\nsis\KYKT Vision Client_0.1.0_x64-setup.exe`
  - `E:\kykt\Coding\4.06\vision_ui\client\src-tauri\target\release\bundle\msi\KYKT Vision Client_0.1.0_x64_en-US.msi`
- The desktop React UI now shows local backend status when it is running inside Tauri, including whether the backend was started by the desktop shell and where the backend log is written.
- 2026-04-11: Desktop UX cleanup pass: the release executable now uses the Windows GUI subsystem to avoid a black console window, the React app waits/retries while the local FastAPI backend starts instead of exposing raw `fetch` errors, and the first screen was simplified into a clean task workbench with advanced parameters folded away by default.
- 2026-04-11: The React/Tauri detail page now behaves more like a task console instead of a file dump. The top area uses a large hero progress block with stage cards, percent, current-phase text, and compact job metadata; outputs are grouped by purpose such as core results, camera/trajectory, masks, confidence arrays, and image visualizations; logs now highlight the latest meaningful line first.
- 2026-04-11: The home page now also acts like a scheduler dashboard. It includes a focus-task overview card, stronger top-level status guidance, job-list filters for all/running/attention/finished, and a model-aware creation guidance box so the first-run path is clearer before testing.
- 2026-04-11: MonST3R dispatch hardening pass: the remote runner defaults were made conservative for first tests (`image_size=224`, `num_frames=24` from the client), `conda run --no-capture-output` is used so stdout reaches the local live log, noisy PyTorch/RoPE warning lines are filtered out of the user-facing progress, and the runner emits heartbeat status while model loading/inference is quiet.
- 2026-04-11: A tiny remote MonST3R smoke run was executed against the uploaded video path. After uploading `third_party/RAFT/models/raft-things.pth`, the smoke run succeeded end-to-end and exported 15 artifacts including `scene.glb`, `pred_traj.txt`, `pred_intrinsics.txt`, confidence maps, dynamic masks, and `scene_meta.json`. Example remote smoke job: `/hdd3/kykt26/jobs/monst3r_smoke_20260411_010843`.
- 2026-04-10: New launch scripts were added:
  - `E:\kykt\Coding\4.06\vision_ui\start_client_rebuild.ps1`
  - `E:\kykt\Coding\4.06\vision_ui\start_desktop_client.ps1`
- FastAPI now exposes JSON endpoints for the rebuild client:
  - `GET /api/bootstrap`
  - `POST /api/jobs`
  - `POST /api/jobs/{job_id}/dispatch`
  - `POST /api/jobs/{job_id}/retry`
  - `POST /api/jobs/{job_id}/duplicate`
  - `POST /api/jobs/{job_id}/cancel`
- Local input caching.
- Automatic filename normalization.
- Arbitrary upload names are accepted.
- Files are stored internally as stable names like `input_01.ext`.
- `job.json` and `status.json` generation.
- Input preview rendering.
- Drag-and-drop upload with file chips and removal.
- Remote dispatch through system `ssh / scp`.
- DUSt3R remote runner upload and execution.
- Multi-image DUSt3R server runner exists under `runners/dust3r_runner.py`.
- DUSt3R jobs now expose multi-image parameters in the local UI:
  - `image_size`
  - `scene_graph`
  - `niter`
  - `lr`
  - `batch_size`
  - `max_points`
- These parameters are stored in `job.json`, uploaded with the job manifest, passed to the remote runner, and written to `scene_meta.json`.
- The UI now offers DUSt3R presets so normal users do not need to manually understand `image_size`; `image_size=512` is the model's internal resize, not a requirement for original uploaded files.
- DUSt3R pair and multi-image jobs share the same frontend flow; N-image behavior is selected by the runner based on input count.
- While a remote runner is active, the local SSH layer polls the remote job `status.json` and maps it into frontend phases.
- Local job JSON updates are guarded with an in-process lock so the log stream and status poller do not write the same `job.json` concurrently.
- Remote launcher uses `conda run -n dust3r` without `--no-banner` for compatibility with the server conda version.
- SSH pipeline execution uses `set -o pipefail` so `tee` does not hide remote command failures.
- Short SSH/SCP operations now use non-interactive OpenSSH options and timeouts to avoid jobs staying in `preparing_remote` forever.
- Required DUSt3R outputs must download successfully before a job is marked finished.
- Static assets are cache-busted through `asset_version` to avoid stale CSS/JS during UI iterations.
- Result download into local cache.
- `scene_meta.json` can be downloaded when produced.
- Browser point cloud preview was removed because large `.ply` files made the detail page sluggish.
- Home page and job detail page poll job state.
- Approximate task progress bars.
- Elapsed time badge on job detail page.
- Live log cache rendering.
- Task actions:
  - run
  - retry current job
  - duplicate as a new job
- Job JSON reads tolerate UTF-8 BOM files created by accidental PowerShell writes.
- Job JSON/status writes are normalized back to plain UTF-8 without BOM.
- UI style is currently a clean light workspace theme.
- The old dark/glass experimental theme was removed.
- Visible mojibake text was cleaned from templates and page scripts.
- 2026-04-06: The visible UI was localized to Chinese end-to-end.
- Chinese localization covers index/detail templates, dynamic JS messages, task status labels, progress text, SSH runner progress messages, and DUSt3R runner status messages.
- UTF-8/mojibake scan passed for templates, static page scripts, app entry, job store, SSH runner, and DUSt3R runner.
- 2026-04-07: Output presentation was refined. The task artifact grid now focuses on core results and hides JSON/log files there; logs remain in the dedicated live log section.
- Point cloud cards now support an enlarged in-page modal viewer, `.ply` download, and local default-app opening via the local FastAPI backend.
- DUSt3R match visualization gained `match_viz_count`, exposed in the advanced parameters. The remote runner now draws reciprocal match lines for the first image pair when this value is greater than zero; otherwise it falls back to a view preview.
- A lightweight cancel action was added. It marks a running job as cancelled locally and uses a safer remote process scan that targets only DUSt3R/MonST3R runner processes referencing the remote job directory.
- 2026-04-07: Browser-side point cloud preview was removed because `.ply` parsing still made the page feel sluggish.
- Point cloud cards now only provide two actions: open the `.ply` with the local default application, intended for MeshLab, and download the `.ply`.
- The Three.js/OrbitControls/PLY viewer scripts are no longer loaded by the job detail page.
- 2026-04-10: The home page now includes a delivery overview panel with task totals and a visible checklist of major unfinished items, so the current gap to a real handoff is visible inside the product itself.
- 2026-04-10: Finished jobs now automatically generate `result_summary.json` and `result_summary.md` in the local job directory. The detail page also renders a human-readable summary block from that data.
- 2026-04-10: `runners/monst3r_runner.py` was upgraded from deployment-check mode to a real non-interactive runner. It reads the uploaded `job.json`, runs the official MonST3R `demo.py`, and copies GLB / trajectory / depth / confidence / image artifacts into the standard local job output flow.
- 2026-04-10: The FastAPI JSON API and React client now expose MonST3R parameters including `image_size`, `num_frames`, `fps`, `batch_size`, `not_batchify`, `real_time`, and window-wise options.
- 2026-04-10: MonST3R result download now pulls the actual remote `output/` and `logs/` tree instead of assuming DUSt3R-only `matches.png` and `pointcloud.ply` files.
- 2026-04-10: Added `E:\kykt\Coding\4.06\vision_ui\SERVER_PREPARATION.md` to track the server-side DUSt3R optimization path and MonST3R deployment checklist.
- 2026-04-10: MonST3R server preparation advanced: repo uploaded to `/hdd3/kykt26/code/monst3r`, env `monst3r` created, dependencies installed, checkpoints confirmed, and the remaining blocker is the first end-to-end validation run through the client.

Still missing:

- Replacing the current Jinja entry pages with the new React client as the default visible UI.
- A fully portable package that bundles Python/runtime dependencies. The current desktop exe supervises the existing project `.venv`; if the project root is moved, set `KYKT_BACKEND_ROOT` to `E:\kykt\Coding\4.06\vision_ui` or the new backend root.
- End-to-end validation of the newly parameterized DUSt3R multi-image path on the server.
- Stronger remote process cleanup verification after cancellation.
- First end-to-end MonST3R client run with one short video or one small frame sequence, followed by output quality inspection.
- A final MonST3R input/output contract after observing the real `.glb`, trajectory, depth, and confidence artifacts on a few examples.
- Better stuck-process recovery on Windows when an old local uvicorn/ssh process refuses termination.
- Exact server-written progress ingestion for every major phase, instead of only mixed local/remote progress approximation.
- More complete task controls such as one-click rerun presets, clearer recovery hints when a job is partially finished, and richer large-result handling for GLB / point-cloud opening workflows.

## SSH / SCP Setup

Chosen transport:

- system `ssh`
- system `scp`

Local SSH alias:

- `KYKT-UI`

Server target:

- host: `172.17.140.97`
- user: `kykt26`
- remote root: `/hdd3/kykt26`

Frontend assumptions:

- DUSt3R repo exists at `/hdd3/kykt26/code/dust3r-main`.
- DUSt3R weight exists at `/hdd3/kykt26/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`.
- Remote conda env name is `dust3r`.
- MonST3R repo exists at `/hdd3/kykt26/code/monst3r`.
- MonST3R conda env name is `monst3r`.
- MonST3R checkpoints are expected at:
  - `/hdd3/kykt26/code/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`
  - `/hdd3/kykt26/code/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth`
  - `/hdd3/kykt26/code/monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt`
- Old uploaded archives are stored at `/hdd3/kykt26/archive/code_uploads_20260410`.

## Current Remote Job Flow

Current DUSt3R flow:

1. Create local job.
2. Normalize uploaded filenames locally.
3. Create remote job directory under `/hdd3/kykt26/jobs/<job_id>`.
4. Upload normalized inputs.
5. Upload `job.json`.
6. Upload the model-specific runner script to `/hdd3/kykt26/runners`.
7. Run remote inference through SSH.
8. Stream stdout into local live logs.
9. Download output files and logs.

## Near-Term Plan

Priority order:

1. Stabilize the frontend task system.
2. Improve remote observability with richer server-written status.
3. Use the rebuilt client or pure-SSH helper scripts to finish one short MonST3R validation run and inspect returned GLB/depth/trajectory artifacts.
4. Make DUSt3R multi-image jobs more reliable and configurable.
5. Add better result viewing for GLB / trajectory / depth / confidence artifacts after observing real MonST3R outputs.
6. Validate MonST3R as a remote runner for video/frame-sequence tasks.
7. Add stronger task controls such as cancel and parameterized reruns.

## Current Delivery Gaps

These are the main blockers between the current state and a more deliverable tool:

1. DUSt3R multi-image flow needs one full validation run with 3 to 5 images and reviewed outputs.
2. Remote cancellation cleanup is safer than before, but still needs post-cancel verification in a real stuck job.
3. MonST3R runner integration now has better logging/defaults, but it still needs the first complete client-triggered server run and output quality inspection.
4. Result packaging and task reporting are still too manual.
5. Recovery and operator guidance still need to be clearer when something hangs or only partially finishes.

## Guidance for Future Agents

- Read this file first.
- Keep the frontend centered on `job.json`, local cache, and SSH/SCP.
- Do not split DUSt3R pair and multi-image into separate frontend products.
- Expand backend runners instead.
- Preserve current server assumptions unless the user changes them explicitly.

Current most important active repo:

- `E:\kykt\Coding\4.06\vision_ui`
