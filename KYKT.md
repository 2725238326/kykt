# KYKT Project Sync

Last updated: 2026-04-10

## Project Lines

This workspace currently has four related but separate 3D vision lines:

- `MVSNet` on `DTU`: supervised multi-view stereo reconstruction. Main outputs are depth maps and fused `.ply` point clouds.
- `SfMLearner` on `KITTI`: self-supervised monocular depth. Main output is KITTI depth metrics and checkpoints.
- `DUSt3R`: pointmap-based 3D reconstruction from image pairs or image sets. Main outputs are match visualizations, point clouds, poses, and focals.
- `MonST3R`: next-stage dynamic/video reconstruction model. Server repo, env, and Python dependencies are prepared, but pretrained checkpoints are not fully downloaded yet.

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
- Official checkpoint download is still blocked by slow or unreliable Google Drive / Dropbox access.
- A residual interrupted `download_ckpt.sh` / `wget models.zip` process was found and stopped on 2026-04-10.
- The temporary local upload area `E:\kykt\Coding\4.10` was cleaned up on 2026-04-10.
- The local MonST3R tarball was deleted after successful server extraction.
- Server upload archives were moved out of `/hdd3/kykt26/code` into `/hdd3/kykt26/archive/code_uploads_20260410`.
- Local helper script for uploading manually downloaded MonST3R weights: `E:\kykt\tools\upload_monst3r_weights.ps1`.
- Local weight staging directory, ignored by git: `E:\kykt\model_uploads\monst3r`.

Notes:

- DUSt3R supports both image pairs and image sets.
- The frontend should not split pair and multi-image into separate products.
- Backend runners should handle whether a job has 2 images or N images.

## Local Frontend

Active project:

- `E:\kykt\Coding\4.06\vision_ui`

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
- A lightweight cancel action was added. It marks a running job as cancelled locally and tries to `pkill` the remote process by matching the remote job directory string.
- 2026-04-07: Browser-side point cloud preview was removed because `.ply` parsing still made the page feel sluggish.
- Point cloud cards now only provide two actions: open the `.ply` with the local default application, intended for MeshLab, and download the `.ply`.
- The Three.js/OrbitControls/PLY viewer scripts are no longer loaded by the job detail page.
- 2026-04-10: The home page now includes a delivery overview panel with task totals and a visible checklist of major unfinished items, so the current gap to a real handoff is visible inside the product itself.
- 2026-04-10: Finished jobs now automatically generate `result_summary.json` and `result_summary.md` in the local job directory. The detail page also renders a human-readable summary block from that data.
- 2026-04-10: A local `monst3r_runner.py` preparation skeleton now exists. It does not run real inference yet, but it can return a clear deployment-preparation message instead of failing as a black box.
- 2026-04-10: Added `E:\kykt\Coding\4.06\vision_ui\SERVER_PREPARATION.md` to track the server-side DUSt3R optimization path and MonST3R deployment checklist.
- 2026-04-10: MonST3R server preparation advanced: repo uploaded to `/hdd3/kykt26/code/monst3r`, env `monst3r` created, dependencies installed, and the remaining blocker is checkpoint acquisition.

Still missing:

- End-to-end validation of the newly parameterized DUSt3R multi-image path on the server.
- Stronger remote process cleanup and verification after cancellation.
- MonST3R checkpoint acquisition, one official demo run, and then a real remote inference runner with a final input/output contract.
- Better stuck-process recovery on Windows when an old local uvicorn/ssh process refuses termination.
- Exact server-written progress ingestion for every major phase, instead of only mixed local/remote progress approximation.
- More complete task controls such as one-click rerun presets, richer retry guidance, and clearer recovery hints when a job is partially finished.

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
- MonST3R checkpoints are not ready yet.
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
3. Make DUSt3R multi-image jobs more reliable and configurable.
4. Add better browser-side point cloud/result viewing.
5. Integrate MonST3R as a new remote runner for video/frame-sequence tasks.
6. Add stronger task controls such as cancel and parameterized reruns.

## Current Delivery Gaps

These are the main blockers between the current state and a more deliverable tool:

1. DUSt3R multi-image flow needs one full validation run with 3 to 5 images and reviewed outputs.
2. Remote cancellation and cleanup still need to become more reliable.
3. MonST3R is still only planned, not truly integrated.
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
