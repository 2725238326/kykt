# Active 3R Model Download / Upload Plan

Last updated: 2026-04-21

## Scope

This document is a plan only. Do not download, upload, clone, or install from this document unless the user explicitly asks to execute a specific model setup step.

Active setup candidates:

1. Spann3R
2. Align3R
3. Fast3R
4. CUT3R

Existing deployed baselines:

- MASt3R: `/hdd3/kykt26/code/mast3r`
- MonST3R: `/hdd3/kykt26/code/monst3r`

## Shared Rules

Local staging root:

```text
E:\kykt\model_uploads\active_3r
```

Remote code root:

```text
/hdd3/kykt26/code
```

Remote archive root:

```text
/hdd3/kykt26/archive/code_uploads_3r
```

Remote model/cache root:

```text
/hdd3/kykt26/models
```

General upload options:

1. **Preferred for full repos**: clone/download locally, compress to `.tar.gz`, upload with `scp`, extract remotely.
2. **Preferred for large weights**: download manually or with Hugging Face / gdown locally, then upload to a model-specific checkpoint folder.
3. **Alternative**: use SFTP client such as Electerm when browser-based downloads are easier.
4. **Avoid**: mixing model-specific CUDA extensions across repos. Keep every model in its own conda env.

General remote checks:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\tools\check_3r_remote.ps1 -SshAlias KYKT-UI
powershell -ExecutionPolicy Bypass -File E:\kykt\tools\check_3r_remote.ps1 -SshAlias KYKT-UI -Json
```

## Spann3R

Purpose:

- DUSt3R-family spatial memory / global pointmap reconstruction.

Official sources:

- Repo: `https://github.com/HengyiWang/spann3r`
- Project page: `https://hengyiwang.github.io/projects/spanner`
- Paper: `https://arxiv.org/abs/2408.16061`

Recommended environment from official docs:

- Python 3.9
- CUDA 11.8
- PyTorch 2.3.0
- torchvision 0.18.0
- torchaudio 2.3.0

Local staging:

```text
E:\kykt\model_uploads\active_3r\spann3r\
  repo\
  checkpoints\
  archives\
```

Remote target:

```text
/hdd3/kykt26/code/spann3r
```

Weights:

- DUSt3R base checkpoint:
  `https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- Spann3R checkpoint folder:
  `https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing`

Download plan:

1. Clone official repo locally or directly on server if GitHub access is stable.
2. Download DUSt3R base checkpoint with `wget` or browser.
3. Download Spann3R checkpoint from Google Drive manually or via `gdown` if folder access works.
4. Keep raw downloads under local `checkpoints\` until the exact filenames are known.

Upload plan:

```powershell
scp E:\kykt\model_uploads\active_3r\spann3r\archives\spann3r_repo.tar.gz KYKT-UI:/hdd3/kykt26/archive/code_uploads_3r/
scp E:\kykt\model_uploads\active_3r\spann3r\checkpoints\*.pth KYKT-UI:/hdd3/kykt26/code/spann3r/checkpoints/
```

Remote smoke command from official docs:

```bash
python demo.py --demo_path ./examples/s00567 --kf_every 10 --vis --vis_cam
```

Notes:

- Compile `croco/models/curope`.
- Watch Open3D version. Official notes mention Open3D 0.16.0 issues.
- Use `--save_ori` if Nerfstudio / NeRF / 3D Gaussian compatible transforms are needed.

## Align3R

Purpose:

- Dynamic video depth consistency, dynamic point clouds, and camera poses.

Official sources:

- Repo: `https://github.com/jiah-cloud/Align3R`
- Paper: `https://arxiv.org/abs/2412.03079`

Recommended environment from official docs:

- Python 3.11
- PyTorch with CUDA 12.1 install path

Local staging:

```text
E:\kykt\model_uploads\active_3r\align3r\
  repo\
  checkpoints\
  depth_pro\
  depth_anything_v2\
  raft\
  archives\
```

Remote target:

```text
/hdd3/kykt26/code/align3r
```

Weights:

- Depth Pro model:
  `https://huggingface.co/cyun9286/Align3R_DepthPro_ViTLarge_BaseDecoder_512_dpt`
- Depth Anything V2 model:
  `https://huggingface.co/cyun9286/Align3R_DepthAnythingV2_ViTLarge_BaseDecoder_512_dpt`
- Google Drive fallback:
  `https://drive.google.com/file/d/1-qhRtgH7rcJMYZ5sWRdkrc2_9wsR1BBG/view?usp=sharing`
  `https://drive.google.com/file/d/1PPmpbASVbFdjXnD3iea-MRIHGmKsS8Vh/view?usp=sharing`

Download plan:

1. Clone official repo.
2. Pull Hugging Face weights using `huggingface-cli download` when network is stable.
3. If Hugging Face is slow, use the Drive fallback with browser or `gdown --fuzzy`.
4. Download external Depth Pro / Depth Anything V2 dependencies according to the official README.

Upload plan:

```powershell
scp E:\kykt\model_uploads\active_3r\align3r\archives\align3r_repo.tar.gz KYKT-UI:/hdd3/kykt26/archive/code_uploads_3r/
scp E:\kykt\model_uploads\active_3r\align3r\checkpoints\* KYKT-UI:/hdd3/kykt26/code/align3r/checkpoints/
```

Remote smoke command from official docs:

```bash
bash demo.sh
```

Notes:

- Requires extra Depth Pro / Depth Anything V2 setup.
- Compile `curope`.
- Some point-cloud visualization relies on MonST3R `viser` utilities.
- The runner contract should export `depth_*.png`, dynamic point cloud, camera poses, and `scene_meta.json`.

## Fast3R

Purpose:

- Fast feed-forward reconstruction for medium/large image collections.

Official sources:

- Repo: `https://github.com/facebookresearch/fast3r`
- Model: `https://huggingface.co/jedyang97/Fast3R_ViT_Large_512`
- Paper: `https://arxiv.org/abs/2501.13928`

Recommended environment from official docs:

- Python 3.11
- CUDA 12.4 example install path
- PyTorch install via official conda command in README

Local staging:

```text
E:\kykt\model_uploads\active_3r\fast3r\
  repo\
  hf_cache\
  checkpoints\
  archives\
```

Remote target:

```text
/hdd3/kykt26/code/fast3r
```

Weights:

- Hugging Face:
  `jedyang97/Fast3R_ViT_Large_512`

Download plan:

1. Clone official repo.
2. Pre-cache Hugging Face model locally or on server:
   `huggingface-cli download jedyang97/Fast3R_ViT_Large_512`
3. If server internet is unreliable, download locally and upload HF cache/checkpoint files.

Upload plan:

```powershell
scp E:\kykt\model_uploads\active_3r\fast3r\archives\fast3r_repo.tar.gz KYKT-UI:/hdd3/kykt26/archive/code_uploads_3r/
scp -r E:\kykt\model_uploads\active_3r\fast3r\hf_cache\* KYKT-UI:/hdd3/kykt26/models/fast3r/
```

Remote smoke command from official docs:

```bash
python fast3r/viz/demo.py
```

Notes:

- Do **not** install DUSt3R `cuROPE` in the Fast3R environment.
- First run may try to download model config/weights from Hugging Face.
- Runner should start with 20 images before testing 50/100 images.

## CUT3R

Purpose:

- Online / persistent-state 3D reconstruction baseline.

Official sources:

- Repo: `https://github.com/CUT3R/CUT3R`
- Project page: `https://cut3r.github.io/`
- Paper: `https://arxiv.org/abs/2501.12387`

Recommended environment from official docs:

- Python 3.11
- PyTorch with CUDA 12.1 install path

Local staging:

```text
E:\kykt\model_uploads\active_3r\cut3r\
  repo\
  checkpoints\
  archives\
```

Remote target:

```text
/hdd3/kykt26/code/cut3r
```

Weights:

- `cut3r_224_linear_4.pth`:
  `https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link`
- `cut3r_512_dpt_4_64.pth`:
  `https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link`

Download plan:

1. Clone official repo.
2. Download checkpoints with browser or `gdown --fuzzy`.
3. Keep both 224 and 512 checkpoints if disk allows; start smoke with 512 if memory permits, otherwise 224.

Upload plan:

```powershell
scp E:\kykt\model_uploads\active_3r\cut3r\archives\cut3r_repo.tar.gz KYKT-UI:/hdd3/kykt26/archive/code_uploads_3r/
scp E:\kykt\model_uploads\active_3r\cut3r\checkpoints\*.pth KYKT-UI:/hdd3/kykt26/code/cut3r/src/
```

Remote smoke command from official docs:

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp
```

Notes:

- Requires `curope`, `gsplat`, `evo`, and `open3d`.
- Official README mentions `llvm-openmp<16`.
- Memory grows roughly linearly with frame count, so smoke tests should start short.

## Upload Verification

After any future upload, run:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\tools\check_3r_remote.ps1 -SshAlias KYKT-UI
```

For structured output:

```powershell
powershell -ExecutionPolicy Bypass -File E:\kykt\tools\check_3r_remote.ps1 -SshAlias KYKT-UI -Json
```

