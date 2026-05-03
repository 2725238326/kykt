# KYKT 工作历程与当前路线

Last updated: 2026-05-03

## 1. 路线变化

早期路线是课程复现与基础实验：

- MVSNet / DTU
- SfMLearner / KITTI
- DUSt3R 静态重建

中期路线是 3R / visual geometry model workbench（多模型重建工作台）。

**当前主线已经迁到 `E:\kykt\Dream`**：架构优先的 3R / 空间智能研究引擎。`Coding\4.06\vision_ui` 现在是支撑研究的工程层，不再是工作重心。

Dream 当前阶段：`Phase 1.5: Research Workflow Deployment`，卡在用户选择 `Dream\planning\BRANCH_SHORTLIST_DECISION_SURFACE.md` 中的 finalist 分支（A/B/C/D）。在用户决定前不进入任何具体 thesis 或机制 spec。

工程层主动模型不变：

1. MASt3R
2. MonST3R
3. Spann3R
4. Align3R （curope 已修，待写 runner）
5. Fast3R
6. CUT3R （curope 已修，待写 runner）

暂缓：DUSt3R multi-image / Pi3X / ZipMap / LingBot-Map。

## 2. 已完成结果

### MASt3R

- 平台 smoke job：`20260420-222729`
- 输出：`matches.png`、`pointcloud.ply`、`scene_meta.json`、`runner.log`
- 点云：301905 个原始点下采样到 250000 个点

### MonST3R

- 标准视频 job：`20260420-222928`
- 参数：`image_size=512`、`num_frames=48`、`window_wise=false`
- 输出：`scene.glb`、`pred_traj.txt`、`pred_intrinsics.txt`、48 张帧预览、96 张动态 mask、96 个 confidence array

### Spann3R

- 服务器目录：`/hdd3/kykt26/code/spann3r`，env：`spann3r`
- 权重：`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`、`spann3r.pth`、`spann3r_101.pth`
- `curope` 已按 TITAN RTX / sm75 编译成功，官方 `s00567` smoke 通过
- 平台 E2E：`20260425-113227`（输入用 6 张 MonST3R 帧预览）

### Fast3R

- 服务器目录：`/hdd3/kykt26/code/fast3r`，env：`fast3r`
- 权重：`/hdd3/kykt26/models/fast3r/Fast3R_ViT_Large_512/model.safetensors`
- TITAN RTX 必须用 `attention_backend=pytorch_naive`，已写进 runner 默认
- 平台 E2E：`20260425-113002`，回传 `pointcloud.ply`、`camera_poses.json`、`confidence_summary.json`、`metadata.json`、`scene_meta.json`

### Align3R （2026-05-03 解锁）

- 服务器目录：`/hdd3/kykt26/code/align3r`，env：`align3r`
- 权重已上传：`DepthPro/model.safetensors`、`DepthAnythingV2/model.safetensors`
- **旧文档说的"系统 CUDA 11.3 vs torch cu121 mismatch"是误诊**，实际系统是 `/usr/local/cuda-12.6/bin/nvcc`，torch cu121 forward-compat 没问题。
- **真正阻塞点是 GLIBC**：env 里预装的 `croco/models/curope/curope.cpython-311-*.so` 需要 `GLIBC_2.32`，宿主 libc 太老。
- **修复**：在 align3r env 内原地重新编译 curope（`TORCH_CUDA_ARCH_LIST=7.5`、`CUDA_HOME=/usr/local/cuda-12.6`），新 .so 时间戳 `5月 3 19:01`，`cuRoPE2D` 可实例化、CUDA kernel 可被调用。
- 仍缺：`align3r_runner.py`、第一个平台 smoke。

### CUT3R （2026-05-03 解锁）

- 服务器目录：`/hdd3/kykt26/code/cut3r`，env：`cut3r`
- 权重：`cut3r_224_linear_4.pth`、`cut3r_512_dpt_4_64.pth`
- **真正阻塞点是 curope 从未编译过**：`src/croco/models/curope/` 没有 build 目录，没有 .so。
- **修复**：同样的 in-place 编译流程，新 .so 时间戳 `5月 3 19:23`，`cuRoPE2D` 可实例化、CUDA kernel 可被调用。
- 仍缺：`cut3r_runner.py`、官方 `examples/001` 的平台 smoke。

## 3. 平台层最近修复（2026-05-03）

| 项目 | 改动文件 | 说明 |
|---|---|---|
| 默认前端切到 React | `Coding/4.06/vision_ui/app.py` | `client/dist/index.html` 存在时，`/` 与 `/jobs/{id}` 直接 serve React，并把 `client/dist/assets/` 挂到 `/assets`。Jinja 模板作为降级保留。 |
| Tauri 端可移植 | `Coding/4.06/vision_ui/client/src-tauri/src/lib.rs` | `is_backend_root` 不再要求 `.venv`。Python 解释器解析顺序：`KYKT_BACKEND_PYTHON` → `<root>/.venv/Scripts/python.exe` → `<root>/python/python.exe` → 相邻 `python/` → 系统 PATH。 |
| Portable bundle 文档 | `Coding/4.06/vision_ui/PORTABLE_BUNDLE.md` | 嵌入式 Python 打包步骤、目录布局、检查清单。 |
| 远端取消硬化 | `Coding/4.06/vision_ui/ssh_runner.py` | `_kill_remote_job_processes` 增加 `align3r_runner.py / cut3r_runner.py / run_job.py`，SIGTERM → 2s grace → SIGKILL → 1s verify，返回 `{killed, remaining}`，按 PID 报告残留进程，并写 `logs/dispatch.debug.log`。 |
| 孤儿任务自愈 | `Coding/4.06/vision_ui/job_store.py`、`app.py` | 新增 `recover_orphan_running_jobs()`，FastAPI startup hook 把后端重启后残留的 `status="running"` 任务标 `failed` 并附重试提示。 |
| 远端 curope 编译辅助 | `tools/build_curope.sh`（新）、`tools/probe_curope.py`、`tools/verify_curope2.py`（暂存于 `tmp/`） | 后续如需对其他 env 重编可复用同一套脚本。 |

## 4. App 当前能力

- 模型路线面板 / 样例库 / 测评矩阵
- 任务人工评分 + AI Advisor 报告（OpenAI 兼容端点）
- 任务详情结果分组（核心 3D / 相机轨迹 / 深度 / 动态 mask / 置信 / 帧预览）
- 一键导出 ZIP bundle（`/api/jobs/{id}/bundle`）
- Tauri 自动监管 FastAPI（`127.0.0.1:8765`），日志 `local_jobs/_desktop/backend.log`
- 远端部署摘要（`/api/deployment/status`）
- **新**：默认 React UI；浏览器直连或桌面端都同一界面
- **新**：取消任务时按 PID 上报清理结果
- **新**：后端重启后自动洗掉假在跑的任务

关键 API：

- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/samples`
- `GET /api/deployment/status`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/evaluation`
- `POST /api/jobs/{job_id}/evaluation`
- `POST /api/jobs/{job_id}/cancel`（清理结果详尽报告）

## 5. 主要 Gap

1. Align3R / CUT3R runner 与第一个平台 smoke。
2. MASt3R / MonST3R / Spann3R / Fast3R 的更高质量样例（用于横向对比矩阵）。
3. 按 `sample_id` 聚合的模型对比视图。
4. portable bundle 一键发布脚本（嵌入式 Python + React dist + exe + backend）打包测试。
5. Windows 侧旧 uvicorn/ssh 进程的恢复仍然偏手动。
6. 服务器端写入的精确进度替换当前的本地+远端混合估算。

## 6. 后续 agent 工作指引

后续 agent 必须先读：

1. `KYKT.md`
2. `PROJECT_PROGRESS_2026-05-03.md`（本文件）
3. `PROJECT_PROGRESS_2026-04-23.md`（历史）
4. `Dream\AGENT_MASTER_PROMPT.md` —— **如果用户的请求涉及研究/新方向/论文，先按这份执行其 mandatory load protocol**
5. `Coding\4.06\vision_ui\ACTIVE_MODEL_INTEGRATION_PLAN.md`
6. `Coding\4.06\vision_ui\MODEL_DEPLOYMENT_STATUS.md`
7. `Coding\4.06\vision_ui\APP_ARCHITECTURE_OPTIMIZATION.md`
8. `Coding\4.06\vision_ui\PORTABLE_BUNDLE.md`（如要碰桌面打包）

工程优先级：

1. 写 `align3r_runner.py` 与 `cut3r_runner.py`。
2. 跑两条 first smoke 并核对输出合同。
3. 设计模型对比聚合视图。
4. 准备 portable bundle 一键发布脚本。

研究优先级：

1. 按 Dream `WORKFLOW_STATUS.md` 的指引等待用户对 finalist 分支的选择，**不要越权进入 thesis spec**。

## 7. 后续 agent 审查要求

后续 agent 不应默认相信现有结论，应主动审查：

- MASt3R / MonST3R / Spann3R / Fast3R 的结果是否真的适合作为基线
- Spann3R 输出的 PLY / NPY / transforms 是否足够接入统一 result contract
- Fast3R 用 `pytorch_naive` 后性能是否还能接受
- 旧 4-23 文档里 Align3R/CUT3R 的"CUDA 11.3 mismatch"诊断已被本次推翻，确认相关 README_SETUP.md 和接入计划同步过来
- app 里模型状态是否与服务器真实状态一致
- 本地和服务器路径是否有重复、过期或错误引用
- React 默认 UI 切换后，浏览器直接访问 localhost:8765 与 Tauri 桌面端行为是否一致

审查时必须使用具体文件和任务 ID，不要只给泛泛意见。
