# 3R 模型接入与研究路线

Last updated: 2026-04-20

## 1. 总定位

`vision_ui` 后续不应绑定单一模型，而应发展成一个 3R/视觉几何模型实验平台。MonST3R 是视频/动态场景方向的重要主线之一，但不是绝对核心。平台核心应是统一输入、统一远端 runner、统一输出合同、统一可视化和统一评估报告。

当前模型分层：

1. **已有基础线**：DUSt3R、MASt3R、MonST3R。
2. **DUSt3R 直接改进线**：Spann3R、Fast3R、MUSt3R、Pow3R、Speed3R。
3. **视频/动态深度与重建线**：MonST3R、Align3R、Easi3R、Geo4D。
4. **状态/流式重建线**：Spann3R、CUT3R、ZipMap、LingBot-Map、LONG3R。
5. **通用视觉几何基础模型线**：VGGT、Pi3/Pi3X、MapAnything。
6. **后处理与展示线**：点云/GLB 清理、轨迹可视化、关键帧预览、mask/置信图解释、结果报告。

## 2. 接入优先级

### P0：先稳住现有平台合同

目标：无论接入哪个模型，都复用同一套平台结构。

- `job.json`：记录模型、输入类型、参数、备注。
- `status.json`：记录远端阶段、进度、错误。
- `output/scene_meta.json`：记录模型输出合同和语义分组。
- `result_summary.json / result_summary.md`：记录结果摘要、核心检查对象、后续动作。
- React 结果页：按核心结果、相机/轨迹、帧预览、mask、置信数组、其他文件分组展示。

验收标准：

- 新模型接入时只需要新增 `model_registry` 条目、runner、SSH dispatch 分支和输出分组规则。
- 不再为每个模型写一套独立 UI。

### P1：现有三条线补齐

#### DUSt3R

研究价值：基础 pairwise pointmap 范式，所有 3R 方向的起点。

工程任务：

- 完成 3 到 5 张图的多图验证。
- 固化 `complete` 与 `swin-5` 的使用边界。
- 输出 `matches.png / pointcloud.ply / scene_meta.json` 的稳定合同。

#### MASt3R

研究价值：DUSt3R 的静态多图匹配增强线。

工程任务：

- 完成服务器 smoke test。
- 复用 DUSt3R 输出合同。
- 与 DUSt3R 做同一组图片对比。

#### MonST3R

研究价值：视频/动态场景重建线。

工程任务：

- 固定 3 条标准测试视频。
- 比较 `224/512`、`24/48/72/96` 帧、window-wise 开关。
- 强化 GLB、轨迹、帧预览、dynamic mask、confidence 的展示与报告。

### P2：第一批新增模型

#### Spann3R

定位：DUSt3R + spatial memory。它直接把每张图的 pointmap 放到全局坐标系，减少或避免 DUSt3R 多图 global alignment。

为什么优先：

- 和 DUSt3R 关系最直接，适合解释“从 pairwise 到 memory/global pointmap”。
- 官方有 demo、Gradio、checkpoint。
- 输出形态和我们当前点云/轨迹/帧序列合同接近。

接入难度：中等。

风险：

- 依赖栈偏旧：README 推荐 PyTorch 2.3.0 + CUDA 11.8。
- demo 有交互可视化逻辑，需要整理成非交互 runner。

第一步：

- 在服务器新建 `spann3r` env。
- 跑官方 example。
- 写 `spann3r_runner.py`，输出 `pointcloud.ply`、相机/帧元数据、`scene_meta.json`。

#### Fast3R

定位：多图一前向，目标是 1000+ images 级别快速重建。

为什么优先：

- 官方提供明确的 Python inference 示例。
- 输入支持图片和视频 demo。
- 适合对比 DUSt3R/MASt3R 在长图集上的速度和结果。

接入难度：中等。

风险：

- 官方提醒不要安装 DUSt3R 的 cuROPE，否则会影响结果。
- 国内 Hugging Face 下载可能不稳定，需要预下载权重。
- 可能遇到 FlashAttention 兼容问题，需要回退 PyTorch attention。

第一步：

- 新建 `fast3r` env，不复用 dust3r/mast3r env。
- 跑 5 张图、20 张图、短视频抽帧三个 smoke case。
- 输出 `pointcloud.ply`、camera poses、confidence summary。

#### Pi3/Pi3X

定位：reference-free / permutation-equivariant visual geometry。Pi3X 支持相机位姿、内参、深度等条件注入，适合做通用视觉几何基础模型对比。

为什么优先：

- 命令行推理清楚，支持图片目录和视频。
- 输出含 point cloud、camera poses、confidence。
- Pi3X 的条件注入很适合后续利用 COLMAP/DUSt3R/MonST3R 的中间结果做融合实验。

接入难度：低到中。

风险：

- 权重是非商业研究许可，要在文档中标清。
- 长序列显存和速度需要实测。

第一步：

- 跑 `example_mm.py --data_path <image_or_video> --save_path result.ply`。
- 接入 `pi3x_runner.py`，输出点云、相机、置信摘要。

#### Align3R

定位：面向动态视频的一致性单目深度、动态点云和相机位姿估计。它利用外部单目深度模型（Depth Pro 或 Depth Anything V2）增强 DUSt3R，再通过全局对齐让不同时间步的深度、相机和点云一致。

为什么优先：

- 和 MonST3R 的输入/输出非常接近，适合做同一段视频的正面对比。
- 它的研究问题更聚焦于“视频深度时间一致性”，能补上 MonST3R 之外的动态视频深度评估角度。
- 官方代码、权重和 Hugging Face demo 都已释放。

接入难度：中高。

风险：

- 依赖较多：DUSt3R、Depth Pro、Depth Anything V2、RAFT、RoPE cuda kernels。
- 官方 quick start 推荐 PyTorch + CUDA 12.1，并要单独安装第三方深度模型。
- 输出可能更偏 depth / dynamic point clouds / pose，需要先明确如何转成我们标准的 `scene_meta.json` 和展示卡片。

第一步：

- 新建 `align3r` env，不复用 `monst3r`。
- 先跑官方 demo 或脚本，确认能输出视频深度、动态点云和相机位姿。
- 接入 `align3r_runner.py`，输出 depth previews、dynamic point cloud、camera poses、`scene_meta.json`。

### P3：第二批研究型模型

#### CUT3R

定位：continuous updating transformer，persistent state，在线重建和动态场景。

价值：

- 是 MonST3R 之后的视频/动态/流式方向的重要候选。
- 可作为“有状态模型”研究线，与 Spann3R/ZipMap/LingBot-Map 对比。

接入难度：中高。

优先动作：

- 先跑官方 demo。
- 观察输出是否能稳定保存点云、相机和状态结果。

#### LingBot-Map

定位：geometric context transformer for streaming reconstruction。

价值：

- 更接近长期目标中的“流式建图”和机器人/在线场景理解。
- 可与 MonST3R、CUT3R 做视频任务对比。

接入难度：高。

风险：

- 依赖 PyTorch 2.9.1 + CUDA 12.8 + FlashInfer。
- 需要确认 TITAN RTX 24GB 下的可用帧数和速度。

#### ZipMap

定位：linear-time stateful reconstruction via test-time training。

价值：

- 适合讲“长序列、状态压缩、线性复杂度”。
- 可与 VGGT/Fast3R/Pi3X 在长图集上比较。

接入难度：中到中高。

优先动作：

- 先跑 streaming demo。
- 观察是否能导出标准 point cloud / camera / state query 结果。

## 3. 研究问题设计

后续不只是“把模型跑起来”，还要回答这些问题：

1. **pairwise vs global/memory**
   - DUSt3R/MASt3R 依赖 pairwise 和全局对齐。
   - Spann3R/Fast3R/Pi3X 更强调直接多图或全局坐标输出。
   - 研究问题：哪类输入下 global/memory 明显优于 pairwise alignment？

2. **静态 vs 动态**
   - DUSt3R/MASt3R 更适合静态多图。
   - MonST3R/Align3R/CUT3R/LingBot-Map 更适合视频/动态。
   - 研究问题：动态物体比例多大时，静态模型开始明显失败？

3. **几何重建 vs 视频深度一致性**
   - MonST3R 更强调动态场景几何估计和可视化导出。
   - Align3R 更强调时间一致的视频深度、动态点云和相机位姿。
   - 研究问题：同一视频下，哪个模型更适合展示三维场景，哪个模型更适合做逐帧深度/轨迹分析？

4. **短序列 vs 长序列**
   - MonST3R 适合短视频验证。
   - Fast3R/ZipMap/LingBot-Map 更适合长序列探索。
   - 研究问题：帧数从 24 增到 96、200、500 时，质量和耗时如何变化？

5. **输出可解释性**
   - 不能只看点云好不好看。
   - 还要看轨迹、内参、置信度、mask、代表帧和失败原因。
   - 研究问题：哪些指标能提前预测一个重建结果是否适合作为展示样例？

6. **平台化接入成本**
   - 每个模型统计从 clone 到 first smoke run 的时间。
   - 记录依赖冲突、权重下载、显存峰值、输出整理难度。
   - 研究问题：哪些模型适合课程展示，哪些更适合论文阅读，不适合工程接入？

## 4. 两周推进安排

### 第 1 阶段：统一评测样例

时间：1 到 2 天。

任务：

- 固定 3 组图片样例：双图、小多图、长多图。
- 固定 3 组视频样例：短静态、动态物体、困难场景。
- 每个样例写入 `samples_manifest.json`，记录用途、难度、推荐模型。

产出：

- 统一样例库。
- 每个样例的人工评价标准。

### 第 2 阶段：现有模型补齐

时间：2 到 3 天。

任务：

- DUSt3R 多图验证。
- MASt3R smoke run。
- MonST3R 标准参数样例和 24/48/72 帧对比。
- Align3R 作为视频/动态深度候选先完成环境可行性判断。

产出：

- 三条已有线的基线结果。
- 每条线的 `result_summary.md`。

### 第 3 阶段：新增 Spann3R

时间：2 到 3 天。

任务：

- 服务器环境新建。
- 跑官方 demo。
- 接入 runner。
- 与 DUSt3R/MASt3R 做同一图集对比。

产出：

- `spann3r_runner.py`
- Spann3R 对比报告。

### 第 4 阶段：新增 Fast3R 或 Pi3X

时间：2 到 4 天。

选择标准：

- 如果想突出“长图集速度”，先 Fast3R。
- 如果想突出“通用视觉几何/无参考视角”，先 Pi3X。

产出：

- 一个新 runner。
- 一张模型横向对比表。

### 第 5 阶段：视频动态与状态/流式研究预研

时间：3 到 5 天。

任务：

- 读 Align3R、CUT3R、ZipMap、LingBot-Map。
- 各自做官方 demo smoke run。
- 只记录接入可行性和输出合同，不急着进 UI。

产出：

- 状态/流式模型调研报告。
- 下一批接入优先级。

## 5. 平台改造任务

这些任务和具体模型无关，应并行推进：

1. 模型 registry 扩展：
   - 增加 `family`：pairwise、multi_view、video_dynamic、streaming_state。
   - 增加 `runner_status`：planned、smoke_ready、integrated、validated。

2. 输出合同扩展：
   - `scene_meta.json` 增加 `model_family`、`artifact_groups`、`primary_artifacts`、`metrics`。
   - 支持 camera poses、intrinsics、confidence、mask、trajectory、point cloud、GLB 的统一描述。

3. 结果页扩展：
   - 增加“模型对比”视图。
   - 同一输入样例下展示不同模型结果。
   - 支持人工评分：结构完整性、轨迹稳定性、噪声、动态区域处理、展示可用性。

4. 研究报告扩展：
   - 每个任务自动生成模型摘要。
   - 每组样例自动生成横向对比表。
   - 支持教师汇报口径：方法原理、输入输出、效果观察、失败原因、下一步。

## 6. 当前决策

1. 不再把 MonST3R 写成绝对核心。
2. 平台核心改成“3R 模型实验与对比工作台”。
3. MonST3R 是视频/动态方向的主线之一。
4. Spann3R 是下一批最值得优先研究和接入的 DUSt3R 改进线。
5. Fast3R 与 Pi3X 是第一批横向扩展候选。
6. Align3R 应加入视频/动态方向候选，和 MonST3R 做同视频对比。
7. CUT3R、ZipMap、LingBot-Map 暂时作为状态/流式模型预研线。
