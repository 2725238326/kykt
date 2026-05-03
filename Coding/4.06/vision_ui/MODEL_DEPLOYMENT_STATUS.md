# Active 3R Model Deployment Status

Last updated: 2026-05-03

## Scope

This status sheet tracks the active model deployment route for the KYKT Vision app. The current active route is:

1. MASt3R
2. MonST3R
3. Spann3R
4. Align3R
5. Fast3R
6. CUT3R

Deferred frontier research: Pi3X, ZipMap, LingBot-Map.

## Server Layout

Remote code root:

```text
/hdd3/kykt26/code
```

Current active directories:

```text
mast3r/
monst3r/
spann3r/
align3r/
fast3r/
cut3r/
```

Old course reproduction repos:

```text
Old/MVSNet/
Old/SfmLearner-Pytorch-master/
```

## Model Status

| Model | Server dir | Env | Current state | Next action |
| --- | --- | --- | --- | --- |
| MASt3R | `/hdd3/kykt26/code/mast3r` | `mast3r` | Platform smoke passed as job `20260420-222729` | Select better 3-8 image static sample |
| MonST3R | `/hdd3/kykt26/code/monst3r` | `monst3r` | Standard 512/48-frame video sample passed as job `20260420-222928` | Manually inspect GLB/trajectory/frame quality |
| Spann3R | `/hdd3/kykt26/code/spann3r` | `spann3r` | Env ready, `curope` compiled for sm75, official `s00567` smoke passed; platform E2E passed as job `20260425-113227` using MonST3R frame previews | Inspect pointcloud/transforms output and decide the next static multiview sample |
| Align3R | `/hdd3/kykt26/code/align3r` | `align3r` | Env ready, **`curope` rebuilt locally on 2026-05-03** (old prebuilt .so failed with `GLIBC_2.32 not found`; rebuilt against env's torch 2.5.1+cu121 and system CUDA 12.6 / TITAN RTX sm75). Module imports and CUDA kernel is reachable. | Write `align3r_runner.py` and run first platform smoke against MonST3R-style video |
| Fast3R | `/hdd3/kykt26/code/fast3r` | `fast3r` | Env ready, local HF weights loaded; platform E2E passed as job `20260425-113002` after runner fallback for local Fast3R loader and sm75 attention | Inspect pointcloud/camera output and keep fallback explicit in reports |
| CUT3R | `/hdd3/kykt26/code/cut3r` | `cut3r` | Env ready, checkpoints present; **`curope` compiled fresh on 2026-05-03** (no prior build artifact existed). Module imports and CUDA kernel is reachable. | Write `cut3r_runner.py` and run first official `examples/001` demo through the platform |

## Official Setup Notes

| Model | Official repo | Baseline env from official docs | First smoke command | Key risk |
| --- | --- | --- | --- | --- |
| Spann3R | `https://github.com/HengyiWang/spann3r` | Python 3.9, CUDA 11.8, PyTorch 2.3.0 | `python demo.py --demo_path ./examples/s00567 --kf_every 10 --vis --vis_cam` | Requires `croco/models/curope` build; Open3D version sensitivity |
| Align3R | `https://github.com/jiah-cloud/Align3R` | Python 3.11, PyTorch CUDA 12.1 install path | `bash demo.sh` | Requires Depth Pro / Depth Anything V2 / RAFT and `curope` |
| Fast3R | `https://github.com/facebookresearch/fast3r` | Python 3.11, CUDA 12.4 example install path | `python fast3r/viz/demo.py` | Do not install DUSt3R `cuROPE`; HF weight download/cache needed |
| CUT3R | `https://github.com/CUT3R/CUT3R` | Python 3.11, PyTorch CUDA 12.1 install path | `python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp` | Requires `curope`, `gsplat`, `evo`, `open3d`; memory grows with frames |

The official setup notes were collected from primary sources on 2026-04-21. Use each model's isolated conda env; do not mix model-specific CUDA extensions across repos.

## Product/App Status

Implemented:

- `model_catalog` in `/api/bootstrap`
- `/api/samples`
- `samples_manifest.json`
- Workbench Light desktop UI aligned with `DESIGN.md`
- Overview command center
- Create workspace with runnable-vs-catalog distinction
- Jobs split-pane workbench with batch actions, keyboard navigation, and inspector rhythm
- Sample Matrix compare workspace with sort/filter/bulk/report flow
- System deployment console with readiness matrix and next-action cards
- Jobs inspector ZIP bundle export through `/api/jobs/{job_id}/bundle`
- Sample Matrix now maps manifest `seed_job_id` entries into matrix cells; `static_pair_easy` is seeded from MASt3R job `20260420-222729`, and `video_static_short` remains seeded from MonST3R job `20260420-222928`
- `tools/check_3r_remote.ps1` remote deployment checker
- Spann3R first smoke completed at `/hdd3/kykt26/code/spann3r/output/demo/s00567_smoke`
- Fast3R first smoke completed at `/hdd3/kykt26/code/fast3r/output/smoke_static_pair/smoke_summary.json`
- Fast3R platform E2E completed as local job `20260425-113002`; output contract returned `pointcloud.ply`, `camera_poses.json`, `confidence_summary.json`, `metadata.json`, `scene_meta.json`, logs, and result summary
- Spann3R platform E2E completed as local job `20260425-113227` using six MonST3R frame previews
- Server verification after upload: `missing_directories=0`, `missing_conda_envs=0`, `missing_required_files=0`
- 2026-05-03: Align3R `croco/models/curope/curope.cpython-311-x86_64-linux-gnu.so` rebuilt in-place with `TORCH_CUDA_ARCH_LIST=7.5`, `CUDA_HOME=/usr/local/cuda-12.6`. Old prebuilt artifact required `GLIBC_2.32`; new build resolves that. Verified `cuRoPE2D` instantiates and the CUDA kernel runs (kernel-internal shape guard reached) under `conda run -n align3r python /tmp/verify_curope2.py`.
- 2026-05-03: CUT3R `src/croco/models/curope/curope.cpython-311-x86_64-linux-gnu.so` compiled fresh (no prior artifact existed) under `conda run -n cut3r ...` with the same toolchain. Verified `cuRoPE2D` instantiates and the CUDA kernel runs.
- 2026-05-03: FastAPI now serves the React client (`client/dist/index.html`) as the default UI on `/` and `/jobs/{id}` when the build exists, with `client/dist/assets/` mounted at `/assets`. Jinja templates remain as a fallback when the build is absent so existing dev flows do not break.
- 2026-05-03: Tauri shell relaxed `is_backend_root` to require only `app.py + job_store.py`. Python interpreter now resolved by a separate `find_backend_python` chain: `KYKT_BACKEND_PYTHON` env → `<root>/.venv/Scripts/python.exe` → `<root>/python/python.exe` → portable `python/` next to or above the exe → system PATH `python.exe`. Portable bundle layout documented in `PORTABLE_BUNDLE.md`.
- 2026-05-03: Cancel cleanup hardened. `_kill_remote_job_processes` now adds `align3r_runner.py`, `cut3r_runner.py`, and `run_job.py` to its needle list, performs a SIGTERM → 2s grace → SIGKILL → 1s verify cycle, and returns `{"killed": [...], "remaining": [...]}` so the cancel message reports specific PIDs and any survivors. Cleanup outcome is also written to `logs/dispatch.debug.log`.
- 2026-05-03: FastAPI startup now calls `recover_orphan_running_jobs()` to flip any `status="running"` job whose runner thread did not survive a backend restart into `failed` with a recovery hint, so the UI never shows ghost-running cards.

Next app tasks:

1. Inspect Fast3R job `20260425-113002` and Spann3R job `20260425-113227` outputs in the desktop client.
2. Pick better static multiview / medium-collection samples for higher-quality Spann3R and Fast3R comparison.
3. Write `align3r_runner.py` and `cut3r_runner.py` (curope blockers cleared 2026-05-03) and run first platform smoke for each.
4. Validate the new React-only UI on a fresh browser session and confirm no Jinja regressions for create/dispatch/retry/cancel flows.
5. Continue splitting `client/src/App.tsx` and tightening evaluation/report/Advisor contracts.

## Download / Upload Planning

The model download and upload plan is documented separately at:

```text
E:\kykt\Coding\4.06\vision_ui\MODEL_DOWNLOAD_UPLOAD_PLAN.md
```

This plan intentionally does not execute downloads or uploads. It lists staging directories, official sources, upload targets, and verification commands.

## Reminder

Do not spend current active engineering time on DUSt3R multi-image, Pi3X, ZipMap, or LingBot-Map unless the user redirects. They remain reference/deferred research tracks.

Do not repeat local downloads by default. The active 3R repositories, weights, and shared smoke samples have already been uploaded to the server.
