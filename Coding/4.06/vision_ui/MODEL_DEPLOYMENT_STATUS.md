# Active 3R Model Deployment Status

Last updated: 2026-04-25

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
| Spann3R | `/hdd3/kykt26/code/spann3r` | `spann3r` | Env ready, `curope` compiled for sm75, official `s00567` smoke passed | Run full client-driven end-to-end validation and confirm output contract holds |
| Align3R | `/hdd3/kykt26/code/align3r` | `align3r` | Env exists, core deps mostly installed; `curope` compile blocked by local CUDA 11.3 vs torch cu121 mismatch | Keep catalog-visible but blocked until slow path or rebuild path is confirmed |
| Fast3R | `/hdd3/kykt26/code/fast3r` | `fast3r` | Env ready, local HF weights loaded, 2-image forward smoke passed with `pytorch_naive` attention fallback on sm75 | Run full client-driven end-to-end validation and keep fallback explicit |
| CUT3R | `/hdd3/kykt26/code/cut3r` | `cut3r` | Env exists, checkpoints present; demo currently fails in RoPE path without compiled `curope` | Keep catalog-visible but blocked until `curope` / torch-CUDA path is fixed |

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
- `tools/check_3r_remote.ps1` remote deployment checker
- Spann3R first smoke completed at `/hdd3/kykt26/code/spann3r/output/demo/s00567_smoke`
- Fast3R first smoke completed at `/hdd3/kykt26/code/fast3r/output/smoke_static_pair/smoke_summary.json`
- Server verification after upload: `missing_directories=0`, `missing_conda_envs=0`, `missing_required_files=0`

Next app tasks:

1. Run full end-to-end Spann3R / Fast3R jobs through the desktop client.
2. Keep blocked-model deployment state explicit and reusable between backend and frontend.
3. Continue splitting `client/src/App.tsx` into workspace-sized components, hooks, and helper modules.
4. Add job bundle export and keep tightening evaluation/report contracts.

## Download / Upload Planning

The model download and upload plan is documented separately at:

```text
E:\kykt\Coding\4.06\vision_ui\MODEL_DOWNLOAD_UPLOAD_PLAN.md
```

This plan intentionally does not execute downloads or uploads. It lists staging directories, official sources, upload targets, and verification commands.

## Reminder

Do not spend current active engineering time on DUSt3R multi-image, Pi3X, ZipMap, or LingBot-Map unless the user redirects. They remain reference/deferred research tracks.

Do not repeat local downloads by default. The active 3R repositories, weights, and shared smoke samples have already been uploaded to the server.
