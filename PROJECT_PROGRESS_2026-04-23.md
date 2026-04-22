# KYKT 工作历程与当前路线

Last updated: 2026-04-23

## 1. 路线变化

早期路线是课程复现与基础实验：

- MVSNet / DTU
- SfMLearner / KITTI
- DUSt3R 静态重建

当前路线已经切换为：

```text
3R / visual geometry model workbench
```

当前主动模型：

1. MASt3R
2. MonST3R
3. Spann3R
4. Align3R
5. Fast3R
6. CUT3R

暂缓模型：

- DUSt3R multi-image
- Pi3X
- ZipMap
- LingBot-Map

## 2. 已完成结果

### MASt3R

- 平台 smoke job：`20260420-222729`
- 输出：
  - `matches.png`
  - `pointcloud.ply`
  - `scene_meta.json`
  - `runner.log`
- 点云：301905 个原始点下采样到 250000 个点。

### MonST3R

- 标准视频 job：`20260420-222928`
- 参数：
  - `image_size=512`
  - `num_frames=48`
  - `window_wise=false`
- 输出：
  - `scene.glb`
  - `pred_traj.txt`
  - `pred_intrinsics.txt`
  - 48 张 frame preview
  - 96 张 dynamic mask
  - 96 个 confidence array

### Spann3R

- 服务器目录：`/hdd3/kykt26/code/spann3r`
- conda env：`spann3r`
- 权重已上传：
  - `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
  - `spann3r.pth`
  - `spann3r_101.pth`
- `curope` 已按 TITAN RTX / sm75 编译成功。
- 官方 `s00567` first smoke 已通过。
- 输出：
  - `/hdd3/kykt26/code/spann3r/output/demo/s00567_smoke/s00567/s00567_conf0.001.ply`
  - `s00567.npy`
  - `transforms.json`

### Fast3R

- 服务器目录：`/hdd3/kykt26/code/fast3r`
- conda env：`fast3r`
- 权重已上传：
  - `/hdd3/kykt26/models/fast3r/Fast3R_ViT_Large_512/model.safetensors`
- first smoke 已通过，条件是把 attention runtime 切到 `pytorch_naive`。
- 原因：TITAN RTX 是 sm75，默认 Flash/Efficient SDPA kernel 不可用。
- 输出：
  - `/hdd3/kykt26/code/fast3r/output/smoke_static_pair/smoke_summary.json`

### Align3R

- 服务器目录：`/hdd3/kykt26/code/align3r`
- conda env：`align3r`
- 权重已上传：
  - `DepthPro/model.safetensors`
  - `DepthAnythingV2/model.safetensors`
- 当前状态：环境部分就绪。
- 阻塞点：`curope` 编译受系统 CUDA 11.3 与 torch cu121 mismatch 影响。

### CUT3R

- 服务器目录：`/hdd3/kykt26/code/cut3r`
- conda env：`cut3r`
- 权重已上传：
  - `cut3r_224_linear_4.pth`
  - `cut3r_512_dpt_4_64.pth`
- 当前状态：环境存在，但 demo 在 RoPE / `curope` 路径触发 CUDA index assert。
- 不建议继续盲目重跑同一命令，应先修 RoPE / `curope` 兼容路径或换更小/更匹配的输入。

## 3. App 当前能力

本地 app 现在具备：

- 模型路线面板
- 样例库 / 测评矩阵
- 远端部署摘要
- 任务人工评分
- AI 评估配置和报告
- Tauri 桌面壳
- FastAPI 后端
- SSH/SCP 调度

关键接口：

- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/samples`
- `GET /api/deployment/status`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/evaluation`
- `POST /api/jobs/{job_id}/evaluation`

## 4. 本地下载清理

本地曾使用：

```text
E:\kykt\model_uploads\active_3r
```

作为代码、权重、样例的 staging 目录。

2026-04-23 已确认：

- 代码已上传到 `/hdd3/kykt26/code/<model>`
- 权重已上传到对应远端目录
- shared samples 已上传到 `/hdd3/kykt26/data/active_3r`
- 远端检查显示 `missing_directories=0`、`missing_required_files=0`、`missing_conda_envs=0`

因此本地 `active_3r` 可安全删除，以释放约 21.6GB。

## 5. 后续 agent 工作指引

后续 agent 必须先读：

1. `KYKT.md`
2. `PROJECT_PROGRESS_2026-04-23.md`
3. `Coding/4.06/vision_ui/ACTIVE_MODEL_INTEGRATION_PLAN.md`
4. `Coding/4.06/vision_ui/MODEL_DEPLOYMENT_STATUS.md`
5. `Coding/4.06/vision_ui/APP_ARCHITECTURE_OPTIMIZATION.md`

后续优先级：

1. 写 `spann3r_runner.py` 并接入 app。
2. 写 `fast3r_runner.py`，必须包含 `pytorch_naive` attention fallback。
3. 处理 Align3R 的 `curope` / CUDA 兼容问题。
4. 处理 CUT3R 的 RoPE / `curope` 路径问题。
5. 做按 `sample_id` 聚合的模型对比视图。

## 6. 后续 agent 审查要求

后续 agent 不应默认相信现有结论，应主动审查：

- MASt3R / MonST3R 的结果是否真的适合作为基线。
- Spann3R 输出的 PLY / NPY / transforms 是否足够接入统一 result contract。
- Fast3R 使用 `pytorch_naive` 后性能是否还能接受。
- Align3R / CUT3R 是否值得继续修环境，还是应该暂时降级为预研。
- app 里模型状态是否与服务器真实状态一致。
- 本地和服务器路径是否有重复、过期或错误引用。

审查时必须使用具体文件和任务 ID，不要只给泛泛意见。

