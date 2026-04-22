# 主动模型接入与对比测评计划

Last updated: 2026-04-20

## 1. 当前聚焦范围

当前阶段不再把 DUSt3R 多图验证作为主动任务。DUSt3R 继续作为理论和代码基座保留，但近期工程和研究推进集中在以下六条模型线：

1. `MASt3R` smoke test
2. `MonST3R` 标准视频样例
3. `Spann3R` 接入
4. `Align3R` 接入
5. `Fast3R` 接入
6. `CUT3R` 接入

同时并行推进：

- 多模型统一输出呈现
- 同输入多模型对比测评
- 自动摘要和报告生成

`Pi3X / ZipMap / LingBot-Map` 暂时单独列为前沿预研，不进入当前主动接入列表。

## 2. 总体策略

不要把六个模型并行硬接进 UI。正确顺序是：

1. 每个模型先完成服务器官方 demo / smoke run。
2. 记录环境、权重、输入格式、输出格式、显存、耗时和失败点。
3. 再写标准 runner，把输出整理成统一合同。
4. 最后才进前端模型列表和模型对比页。

统一合同：

```text
local_jobs/<job_id>/
  input/
  output/
    scene_meta.json
    pointcloud.ply / scene.glb / depth_*.png / frame_*.png / trajectory.*
  logs/
    runner.log
  result_summary.json
  result_summary.md
```

## 3. 模型接入顺序

### 3.1 MASt3R smoke test

目标：把静态多图增强线补齐，作为 DUSt3R-family 的静态基准。

当前进展：

- 2026-04-20：平台链路 smoke test 已完成，job `20260420-222729`。
- 输入：已有 DUSt3R 双图样例。
- 输出：`matches.png`、`pointcloud.ply`、`scene_meta.json`、`runner.log`。
- 点云：从 301905 个原始点下采样到 250000 个点。

输入：

- 2 到 5 张静态图片
- 同一物体或同一室内角落
- 重叠区域明显

输出合同：

- `matches.png`
- `pointcloud.ply`
- `scene_meta.json`
- `runner.log`

验收标准：

- 能通过平台调度远端 MASt3R。
- 本地能回传点云和匹配图。
- `scene_meta.json` 记录图片数量、参数、点云文件、匹配图文件。
- 前端能按“核心成果 / 图像可视化 / 其他产物”显示。

优先级：已完成第一轮 smoke，下一步是换更合适的 3 到 8 张静态图片做质量样例。

### 3.2 MonST3R 标准视频样例

目标：拿到一组可展示的视频动态重建基线。

当前进展：

- 2026-04-20：标准档平台链路已完成，job `20260420-222928`。
- 参数：`image_size=512`、`num_frames=48`、`window_wise=false`。
- 输出：297 个本地产物，其中包括 1 个 `scene.glb`、48 张帧预览、96 张动态 mask、96 个置信数组、轨迹和内参文件。
- 核心检查对象：`scene.glb`、`pred_traj.txt`、`pred_intrinsics.txt`、`frame_0000.png`、`frame_0024.png`、`frame_0047.png`。

输入：

- 5 到 20 秒短视频
- 有平移视差
- 背景稳定、有纹理
- 动态物体可以出现，但不要占满画面

参数组：

- 快速：`image_size=224`, `num_frames=24`
- 标准：`image_size=512`, `num_frames=48`
- 增强：`image_size=512`, `num_frames=72`

输出合同：

- `scene.glb`
- `pred_traj.txt`
- `pred_intrinsics.txt`
- `frame_*.png`
- `dynamic_mask_*.png`
- `conf_*.npy`
- `scene_meta.json`

验收标准：

- 能通过平台完整跑一次标准档。
- GLB 能打开并看到主体结构。
- 轨迹文件存在且非空。
- 帧预览、mask、confidence 能被前端分组展示。
- 摘要给出核心检查顺序。

优先级：已完成标准档样例，下一步是人工检查 GLB/轨迹/帧预览质量，并准备 24/72 帧对比。

### 3.3 Spann3R 接入

目标：接入 DUSt3R 改进线中最自然的一条，研究 spatial memory / global pointmap。

输入：

- 10 到 50 张有序图片
- 也可用视频抽帧

接入步骤：

1. 新建 `spann3r` env。
2. 下载官方 checkpoint 和 DUSt3R checkpoint。
3. 跑官方 example。
4. 整理非交互 runner，避免可视化窗口阻塞。
5. 输出 `pointcloud.ply`、相机/帧元数据、`scene_meta.json`。

验收标准：

- 服务器 smoke run 成功。
- 平台 runner 能非交互执行。
- 同一组输入能和 MASt3R 做结果对比。

优先级：高，作为第一个新增 3R 模型。

### 3.4 Align3R 接入

目标：建立视频动态深度一致性基线，和 MonST3R 做同视频对比。

输入：

- 与 MonST3R 相同的短视频样例

接入步骤：

1. 新建 `align3r` env。
2. 准备 DUSt3R、Align3R、Depth Pro、Depth Anything V2、RAFT 权重。
3. 跑官方 demo。
4. 确认输出：depth maps、dynamic point cloud、camera poses。
5. 写 `align3r_runner.py`。

输出合同：

- `depth_*.png`
- `pointcloud.ply` 或 dynamic point cloud 文件
- `camera_poses.txt/json`
- `scene_meta.json`
- `runner.log`

验收标准：

- 同一视频能与 MonST3R 对比。
- 能展示逐帧深度和相机位姿。
- 摘要能区分“深度一致性结果”和“三维场景展示结果”。

优先级：高，紧跟 Spann3R。

### 3.5 Fast3R 接入

目标：建立长图集快速重建基线。

输入：

- 20 到 100 张图片起步
- 后续扩展到 200+ 图片

接入步骤：

1. 新建 `fast3r` env。
2. 跑官方 inference 示例。
3. 明确是否需要预下载 Hugging Face 权重。
4. 写 `fast3r_runner.py`。

输出合同：

- `pointcloud.ply`
- camera poses
- confidence summary
- `scene_meta.json`

验收标准：

- 20 张图 smoke run 成功。
- 记录耗时和显存。
- 与 MASt3R/Spann3R 在同一长图集上对比。

优先级：中高。

### 3.6 CUT3R 接入

目标：建立 online / persistent state 方向基线。

输入：

- 视频帧序列
- 稀疏照片集合

接入步骤：

1. 新建 `cut3r` env。
2. 跑官方 demo。
3. 观察是否能稳定导出点云、相机和状态结果。
4. 写 `cut3r_runner.py`。

输出合同：

- pointmaps / pointcloud
- camera parameters
- state / revisiting 相关元数据
- `scene_meta.json`

验收标准：

- 官方 demo smoke run 成功。
- 至少一个视频/帧序列能导出可查看结果。
- 能和 MonST3R / Align3R 做视频方向对比。

优先级：中高，放在 Fast3R 后或并行预研。

## 4. 呈现优化方案

### 4.1 统一结果分组

每个模型输出按以下组展示：

1. 核心 3D 结果：`scene.glb`, `pointcloud.ply`, `mesh`, splat 等。
2. 相机与轨迹：poses、trajectory、intrinsics、focal。
3. 深度与几何：depth maps、pointmaps、geometry arrays。
4. 动态区域：mask、motion maps、dynamic point clouds。
5. 置信与诊断：confidence maps、error maps、logs。
6. 帧预览：input frames、sampled frames、render previews。
7. 其他产物。

### 4.2 每个任务的摘要卡

摘要卡必须回答：

- 本次输入是什么？
- 模型输出了什么？
- 最值得先看的文件是哪几个？
- 这次结果是否适合展示？
- 下一轮应该调输入、调参数，还是换模型？

### 4.3 模型专属解释

不同模型要用不同解释口径：

- MASt3R：匹配质量、点云完整性。
- MonST3R：GLB 场景、轨迹、动态 mask。
- Spann3R：global pointmap、一致性、是否省掉明显对齐问题。
- Align3R：逐帧深度连续性、相机 pose 稳定性。
- Fast3R：长图集速度、点云完整性、显存。
- CUT3R：online state、revisiting、长序列稳定性。

## 5. 模型间对比测评方案

### 5.1 样例集

建立 `samples_manifest.json`，至少包含：

1. `static_pair_easy`：2 张静态图片。
2. `static_multiview_small`：3 到 8 张静态图片。
3. `static_collection_medium`：20 到 50 张图片。
4. `video_static_short`：短视频，场景静态，相机移动。
5. `video_dynamic_short`：短视频，有动态物体。
6. `video_hard_case`：弱纹理、反光、快速运动或遮挡。

### 5.2 对比矩阵

| 样例 | MASt3R | MonST3R | Spann3R | Align3R | Fast3R | CUT3R |
|---|---|---|---|---|---|---|
| static_multiview_small | 必跑 | 可选 | 必跑 | 不跑 | 可选 | 可选 |
| static_collection_medium | 可选 | 不跑 | 必跑 | 不跑 | 必跑 | 可选 |
| video_static_short | 不跑 | 必跑 | 可选抽帧 | 必跑 | 可选抽帧 | 必跑 |
| video_dynamic_short | 不跑 | 必跑 | 可选抽帧 | 必跑 | 不优先 | 必跑 |
| video_hard_case | 不跑 | 必跑 | 可选 | 必跑 | 不优先 | 必跑 |

### 5.3 指标

工程指标：

- 环境搭建耗时
- 权重数量和下载难度
- 单次推理耗时
- GPU 显存峰值
- 输出文件大小
- runner 改造难度

结果指标：

- 结构完整性：1 到 5 分
- 轨迹稳定性：1 到 5 分
- 点云/GLB 噪声：1 到 5 分
- 动态区域处理：1 到 5 分
- 深度连续性：1 到 5 分
- 展示可用性：1 到 5 分

平台指标：

- 是否能非交互运行
- 是否能自动写 `status.json`
- 是否能生成 `scene_meta.json`
- 是否能生成摘要
- 是否能前端预览核心结果

### 5.4 最终输出

每个模型接入完成后至少生成：

- 一份 smoke report
- 一份 result summary
- 一份对比表
- 一组可展示截图或预览
- 一个“推荐使用场景”结论

## 6. 执行顺序

当前推荐顺序：

1. MASt3R smoke test
2. MonST3R 标准视频样例
3. 平台模型对比样例集与 `samples_manifest.json`
4. Spann3R 官方 demo 与 runner
5. Align3R 官方 demo 与 runner
6. Fast3R 官方 demo 与 runner
7. CUT3R 官方 demo 与 runner
8. 前端模型对比视图
9. 自动生成横向评测报告

当前完成：

- [x] MASt3R smoke test：`20260420-222729`
- [x] MonST3R 标准视频样例：`20260420-222928`
- [x] 远端 `spann3r / align3r / fast3r / cut3r` 目录 setup checklist
- [x] 本地 `samples_manifest.json` 初版
- [x] 后端 `model_catalog` 元数据初版
- [x] 后端 `/api/samples`：返回样例清单、样例统计、model catalog
- [x] 本地 `MODEL_DEPLOYMENT_STATUS.md`：记录 active 模型部署状态和下一步
- [x] Spann3R 环境、权重、`curope` 编译和官方 `s00567` smoke
- [x] Fast3R 环境、权重和 2 图 smoke；TITAN RTX 需要 `pytorch_naive` attention fallback
- [ ] Align3R 环境仍需处理 `curope` / CUDA 版本兼容
- [ ] CUT3R 环境仍需处理 `curope` / RoPE path 兼容
- [x] 本地 active_3r 下载内容已上传到服务器；后续默认不要重复下载/上传。

## 7. 暂缓事项

以下事项暂时不占主线：

- DUSt3R 多图验证
- Pi3X 接入
- ZipMap 接入
- LingBot-Map 接入
- 模型训练或 fine-tuning
