# MonST3R 主线推进方案

Last updated: 2026-04-20

## 1. 当前判断

MonST3R 是当前平台主线，不是备用模型。`vision_ui` 后续应优先服务 MonST3R 的输入准备、远端推理、结果整理、结果展示和汇报产出；DUSt3R、MASt3R、ZipMap、LingBot-Map、Pi3/Pi3X 等模型只作为对照或补充能力接入。

当前已经具备的基础：

- 本地平台已经有 `job.json / status.json / output / logs` 任务结构。
- 本地前端已经支持视频和帧序列创建 MonST3R 任务。
- 远端 MonST3R repo、conda env、权重和依赖已经就位。
- `runners/monst3r_runner.py` 已经能以非交互方式调用官方 `demo.py`，并整理 `scene.glb`、`pred_traj.txt`、`pred_intrinsics.txt`、帧图、mask、置信数组等产物。
- 已有一次 smoke run 能回传 GLB、轨迹、内参、深度/置信/mask 等结果。

当前主要问题：

- MonST3R 产物数量多，用户看到的是“文件堆”，不是“重建结果”。
- 远端 runner 对产物语义记录不够细，`scene_meta.json` 还不能直接支撑更好的展示和评估。
- 结果页虽然已有分组，但仍缺一个 MonST3R 专属的检查顺序和效果判断结构。
- 后续正式样例需要系统化输入策略和参数消融，而不是随机上传视频后看运气。

## 2. 今晚推进目标

今晚不扩展新模型，先把 MonST3R 主线往“可展示、可复查、可继续优化”推进。

验收目标：

1. 有一份明确的 MonST3R 主线推进文档，后续工作不再被新论文横向扩展打散。
2. 新一轮 MonST3R 任务的 `scene_meta.json` 能记录产物角色、类别统计和优先检查对象。
3. 本地 `result_summary.json / result_summary.md` 能用 MonST3R 语言解释结果，不只是列出文件。
4. React 结果页能展示“核心检查对象”和“产物类别统计”，让 GLB、轨迹、帧预览、mask、置信数组的关系更清楚。
5. 不破坏 DUSt3R/MASt3R 现有链路。

## 3. 推进路线

### A. 输入质量

短期先不重写抽帧链路，但要把目标定下来：

- 标准样例优先选 5 到 20 秒短视频。
- 相机运动要有平移视差，少用原地旋转。
- 场景要有纹理和稳定背景，少用纯白墙、玻璃、大面积反光。
- 动态物体可以出现，但不要占满画面。
- 先固定 3 条测试视频：短静态场景、有人/物体动态场景、困难场景。

后续平台化目标：

- 上传后自动读取视频时长、分辨率、fps。
- 给出抽帧建议。
- 自动提示模糊、过暗、过长、重复帧风险。

### B. 参数策略

今晚继续沿用当前参数体系，不大改 UI：

- 快速验链路：`image_size=224`, `num_frames=24`
- 标准样例：`image_size=512`, `num_frames=48`, `window_wise=false`
- 增强样例：`image_size=512`, `num_frames=72/96`
- 长视频：`window_wise=true`, `window_size=24/32`, `window_overlap_ratio=0.5`

后续要形成参数消融表，至少比较：

- `224 / 512`
- `24 / 48 / 72 / 96 frames`
- `window_wise=false / true`
- `window_size=24 / 32`

### C. 远端推理与产出合同

今晚优先强化产物语义：

- `scene.glb`：核心三维场景。
- `pred_traj.txt`：相机轨迹。
- `pred_intrinsics.txt`：相机内参。
- `frame_*.png`：彩色帧预览。
- `dynamic_mask_*.png` / `enlarged_dynamic_mask_*.png`：动态区域辅助判断。
- `conf_*.npy` / `init_conf_*.npy`：置信数组。
- `frame_*.npy`：几何/深度数组。

`scene_meta.json` 应记录：

- `artifact_groups`
- `review_targets`
- `frame_preview_count`
- `dynamic_mask_count`
- `confidence_count`
- `trajectory_count`
- `intrinsics_count`

### D. 结果展示

今晚先做低风险展示增强：

- 摘要区增加核心检查对象。
- 摘要区增加类别统计。
- 输出文件继续按核心成果、图像可视化、相机轨迹、mask、置信数组分组。
- 保持 GLB 使用系统默认程序打开，不重新引入浏览器内重型 3D 解析。

后续增强：

- 为 GLB 生成缩略图或一键外部查看指引。
- 将 `pred_traj.txt` 转成轨迹图。
- 把代表帧、mask、置信图做成一组可快速翻看的检查面板。

## 4. 今晚执行清单

1. 新增本文件，固定 MonST3R 主线策略。
2. 修改 `runners/monst3r_runner.py`，让远端 `scene_meta.json` 记录产物角色和检查目标。
3. 修改 `ssh_runner.py`，让本地 `result_summary` 汇总 MonST3R 产物类别和检查顺序。
4. 修改 React 类型和摘要面板，展示核心检查对象与类别统计。
5. 运行 Python 编译检查和前端 build。

## 5. 明天优先事项

1. 用一段短视频跑标准参数：`image_size=512`, `num_frames=48`。
2. 检查 `scene.glb`、`pred_traj.txt`、`frame_*.png`、mask 和置信数组是否符合展示样例要求。
3. 如果结果可用，复制任务分别跑 `24 / 72` 帧对比。
4. 如果 GLB 结构差，优先换输入视频，而不是先动模型。
5. 将观察结果写回 `result_summary.md` 或 AI 评估报告，沉淀为样例库。

