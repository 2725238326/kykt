# KYKT Project Sync

Last updated: 2026-04-20

## Project Lines

This workspace currently has five related but separate 3D vision lines:

- `MVSNet` on `DTU`: supervised multi-view stereo reconstruction. Main outputs are depth maps and fused `.ply` point clouds.
- `SfMLearner` on `KITTI`: self-supervised monocular depth. Main output is KITTI depth metrics and checkpoints.
- `DUSt3R`: pointmap-based 3D reconstruction from image pairs or image sets. Main outputs are match visualizations, point clouds, poses, and focals.
- `MASt3R`: DUSt3R-family static multi-image matching + reconstruction upgrade line. The app is now being wired to treat it as a DUSt3R-like image job with the same output contract.
- `MonST3R`: next-stage dynamic/video reconstruction model. Server repo, env, Python dependencies, and required checkpoints are now in place; the local client is being wired to run official demo-equivalent inference and pull back artifacts.

## Workspace Layout

- `E:\kykt`: project root.
- `E:\kykt\Coding\3.16`: MVSNet work.
- `E:\kykt\Coding\3.23`: SfMLearner work.
- `E:\kykt\Coding\3.30`: DUSt3R work.
- `E:\kykt\Coding\4.06\vision_ui`: current local frontend project.
- `E:\kykt\Coding\external_sources`: local copies of external upstream repos kept for reference or upload preparation.
- `E:\kykt\Coding\4.06\vision_ui\MONST3R_MAINLINE_PLAN.md`: MonST3R video/dynamic reconstruction track plan.
- `E:\kykt\Coding\4.06\vision_ui\THREER_MODEL_ROADMAP.md`: broader 3R model integration and research roadmap.

## Proven Results

### MVSNet

Known result directory:

- `E:\kykt\Coding\3.16\d192_fast16_eval`

Prior server references:

- `/hdd3/kykt26/code/Old/MVSNet/checkpoints/d192_fast16`
- `/hdd3/kykt26/code/Old/MVSNet/outputs/d192_fast16_eval`

### SfMLearner

Server repo:

- `/hdd3/kykt26/code/Old/SfmLearner-Pytorch-master`

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

- `/hdd3/kykt26/code/Old/SfmLearner-Pytorch-master/checkpoints/rectified,10epochs,epoch_size1000,seq5,b8,s2.0/03-29-10:55`

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
- 2026-04-11: MonST3R dispatch hardening pass: the remote runner defaults were made conservative for first tests (`image_size=224`, `num_frames=24` from the client at that stage), `conda run --no-capture-output` is used so stdout reaches the local live log, noisy PyTorch/RoPE warning lines are filtered out of the user-facing progress, and the runner emits heartbeat status while model loading/inference is quiet.
- 2026-04-12: The advanced-parameter UI now uses recommended dropdown choices instead of mostly freeform inputs. DUSt3R and MonST3R both expose curated presets for image size, iterations, scene graph, frame count, window settings, and related options, with the MonST3R client baseline shifted to a more practical formal-sample preset (`image_size=512`, `num_frames=48`, `window_size=24`).
- 2026-04-12: The desktop client now also exposes one-click preset tiers above the advanced parameters: `快速`, `标准`, and `增强`. Selecting a preset fills the full DUSt3R / MonST3R parameter set at once, while manual field edits automatically move the UI into a custom state.
- 2026-04-13: A tiny model registry layer now exists in `E:\kykt\Coding\4.06\vision_ui\model_registry.py`. The immediate goal is to avoid scattering model strings everywhere; adding a future model should now follow the pattern “register model -> add runner -> add SSH dispatch branch” instead of large rewrites.
- 2026-04-13: MASt3R was added to the local app as a DUSt3R-like image model entry. The first version reuses the DUSt3R parameter family and output contract (`matches.png`, `pointcloud.ply`, `scene_meta.json`) so the frontend does not need a second static-image visualization path.
- 2026-04-13: A minimal OpenAI-compatible experiment advisor layer now exists in `E:\kykt\Coding\4.06\vision_ui\advisor.py`, with local config at `E:\kykt\Coding\4.06\vision_ui\settings\advisor.json`. The desktop detail page now exposes an `AI评估` action that can summarize current results, diagnose issues, suggest next steps, and generate short teacher-facing report wording once `base_url`, `api_key`, and `model` are filled in.
- 2026-04-13: The desktop home page was reorganized into a top-menu workspace layout instead of one long stacked screen. The main menu now splits the app into `工作台`, `文件与新建`, `运行与结果`, `AI评估`, and `帮助与系统`, which reduces clutter and makes the next action clearer.
- 2026-04-13: AI evaluation was promoted from a single detail-page button into a first-class workflow element. The app now shows AI readiness in the status strip, provides an `AI 工作台` with usage guidance, surfaces an “AI 评估建议” block inside task detail, and adds direct copy actions for teacher-facing report wording.
- 2026-04-13: AI configuration no longer requires hand-editing `settings/advisor.json`. The backend now exposes `/api/advisor/config` for safe read/write, and the desktop UI includes a small modal for `enabled / base_url / api_key / model`. Existing saved API keys are not echoed back into the form; leaving the key field blank keeps the current secret unchanged.
- 2026-04-13: Fresh desktop builds for this workspace/AI refactor were synced to `E:\kykt\release\kykt_vision_client`, including `kykt_vision_client.exe`, `kykt_vision_client_workspace_ai.exe`, and the updated NSIS installer.
- 2026-04-20: The platform direction was corrected from a MonST3R-only emphasis to a broader 3R / visual geometry model workbench. MonST3R remains one video/dynamic reconstruction track, while DUSt3R, MASt3R, Spann3R, Align3R, Fast3R, Pi3/Pi3X, CUT3R, ZipMap, LingBot-Map, and related models should be treated as planned comparison / integration candidates.
- 2026-04-20: A dedicated MonST3R video/dynamic track plan was added at `E:\kykt\Coding\4.06\vision_ui\MONST3R_MAINLINE_PLAN.md`.
- 2026-04-20: A broader 3R model integration roadmap was added at `E:\kykt\Coding\4.06\vision_ui\THREER_MODEL_ROADMAP.md`, with the active focus narrowed to MASt3R, MonST3R, Spann3R, Align3R, Fast3R, and CUT3R.
- 2026-04-20: A concrete active integration and comparison plan was added at `E:\kykt\Coding\4.06\vision_ui\ACTIVE_MODEL_INTEGRATION_PLAN.md`. DUSt3R multi-image validation is no longer a current priority; Pi3X, ZipMap, and LingBot-Map are kept as separate frontier research items.
- 2026-04-20: Server code layout was reorganized: old MVSNet and SfMLearner repos were moved under `/hdd3/kykt26/code/Old`, and active 3R candidate directories now exist at `/hdd3/kykt26/code/mast3r`, `/hdd3/kykt26/code/monst3r`, `/hdd3/kykt26/code/spann3r`, `/hdd3/kykt26/code/align3r`, `/hdd3/kykt26/code/fast3r`, and `/hdd3/kykt26/code/cut3r`.
- 2026-04-20: MASt3R platform smoke test succeeded as local job `20260420-222729`, returning `matches.png`, `pointcloud.ply`, `scene_meta.json`, and `runner.log`.
- 2026-04-20: MonST3R standard video sample succeeded as local job `20260420-222928` with `image_size=512` and `num_frames=48`, returning 297 local artifacts including one `scene.glb`, trajectory, intrinsics, 48 frame previews, 96 dynamic masks, and 96 confidence arrays.
- 2026-04-20: Active model directories `/hdd3/kykt26/code/spann3r`, `/hdd3/kykt26/code/align3r`, `/hdd3/kykt26/code/fast3r`, and `/hdd3/kykt26/code/cut3r` now each contain a `README_SETUP.md` checklist.
- 2026-04-20: The local app now exposes a broader model catalog with family, runner status, research priority, and active/deferred flags while keeping the visible runnable model list restricted to integrated models.
- 2026-04-20: A first shared sample/evaluation manifest was added at `E:\kykt\Coding\4.06\vision_ui\samples_manifest.json`.
- 2026-04-20: The React app gained a visible model-roadmap panel on the workbench and system pages. It shows runnable models, planned active 3R integrations, and deferred frontier research models so the operator can see the current MASt3R / MonST3R / Spann3R / Align3R / Fast3R / CUT3R plan inside the app.
- 2026-04-21: Backend endpoint `GET /api/samples` was added to expose `samples_manifest.json`, sample status counts, required-model counts, and the model catalog for the upcoming model comparison UI.
- 2026-04-21: Active deployment status was captured in `E:\kykt\Coding\4.06\vision_ui\MODEL_DEPLOYMENT_STATUS.md`.
- 2026-04-21: The React/Tauri desktop app was rebuilt and synced to `E:\kykt\release\kykt_vision_client`, including `kykt_vision_client.exe`, `kykt_vision_client_3r_samples.exe`, and refreshed NSIS/MSI installers.
- 2026-04-20: MonST3R result semantics were strengthened: future `scene_meta.json` and generated summaries can identify core review targets, artifact groups, frame previews, dynamic masks, confidence arrays, trajectory files, and intrinsics files instead of presenting the output as an undifferentiated file list.
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
- 2026-04-11: Root cause of the “stuck at preparing remote directory” report was clarified. The job itself could move forward, but the desktop UI was capable of showing stale progress when the local FastAPI backend disappeared or when port `8765` was occupied by another `uvicorn app:app` process started outside the managed desktop flow.
- 2026-04-11: The desktop shell now starts the backend asynchronously so the window does not freeze during startup, and it now exposes stronger backend supervision commands instead of only a one-time stale status snapshot.
- 2026-04-11: The React client now degrades gracefully when API polling fails, marks the local service as disconnected, and offers explicit “probe/restart local service” recovery actions instead of silently keeping old task cards on screen.
- 2026-04-11: The Tauri layer now distinguishes “TCP port is occupied” from “backend API is actually healthy”, and a manual restart path can reclaim port `8765` from an old conflicting local process.
- 2026-04-11: MonST3R retry validation passed locally through the JSON API. Job `20260411-124006` advanced across remote directory creation, input upload, remote runner upload, and into official MonST3R model loading / pointcloud export stages.

Still missing:

- Replacing the current Jinja entry pages with the new React client as the default visible UI.
- A fully portable package that bundles Python/runtime dependencies. The current desktop exe supervises the existing project `.venv`; if the project root is moved, set `KYKT_BACKEND_ROOT` to `E:\kykt\Coding\4.06\vision_ui` or the new backend root.
- MASt3R still needs a first real server smoke test and one reviewed output sample.
- Stronger remote process cleanup verification after cancellation.
- First end-to-end MonST3R client run with one short video or one small frame sequence, followed by output quality inspection.
- A final MonST3R input/output contract after observing the real `.glb`, trajectory, depth, and confidence artifacts on a few examples.
- First real MASt3R server smoke test using the server repo/env/weights, followed by one client-dispatched MASt3R sample.
- Filling `settings/advisor.json` with a real OpenAI-compatible endpoint and validating one end-to-end AI evaluation response on a finished job.
- Better stuck-process recovery on Windows when an old local uvicorn/ssh process refuses termination.
- Automatic stale-job recovery after a local backend crash. Right now the UI can reconnect and restart the backend, but partially running jobs are still not rehydrated into an explicit “interrupted / safe to retry” state.
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

1. Treat `vision_ui` as a 3R / visual geometry model workbench: keep one job format, one runner contract, one result summary contract, and one output presentation system.
2. Run MASt3R smoke test and produce the static reconstruction baseline.
3. Run one MonST3R standard video sample and inspect GLB / trajectory / frame / mask / confidence outputs.
4. Build a small shared sample set and `samples_manifest.json` for static multi-view, medium image collection, short video, dynamic video, and difficult cases.
5. Add Spann3R as the first new active 3R model runner.
6. Add Align3R as the video/dynamic depth consistency runner and compare it with MonST3R on the same videos.
7. Add Fast3R for long image collections.
8. Add CUT3R for online / persistent-state reconstruction.
9. Add better result viewing for GLB / PLY / trajectory / frame preview / dynamic mask / confidence / depth artifacts across models.
10. Add model-to-model comparison and scoring: runtime, memory, output completeness, structure quality, trajectory stability, dynamic handling, and presentation usability.
11. Keep Pi3X, ZipMap, and LingBot-Map as separate frontier research items for later.

## Current Delivery Gaps

These are the main blockers between the current state and a more deliverable tool:

1. MASt3R still needs a first real server smoke test and reviewed outputs.
2. MonST3R runner integration now has better logging/defaults, but it still needs a standard client-triggered video sample and output quality inspection.
3. Spann3R, Align3R, Fast3R, and CUT3R still need environment setup, official smoke runs, runner contracts, and UI registration.
4. Cross-model presentation is still too file-list oriented; it needs model-aware grouping and comparison views.
5. Result packaging, scoring, and report generation are still too manual.
6. Recovery and operator guidance still need to be clearer when something hangs or only partially finishes.

## Guidance for Future Agents

- Read this file first.
- Keep the frontend centered on `job.json`, local cache, and SSH/SCP.
- Do not split DUSt3R pair and multi-image into separate frontend products.
- Expand backend runners instead.
- Preserve current server assumptions unless the user changes them explicitly.

Current most important active repo:

- `E:\kykt\Coding\4.06\vision_ui`
