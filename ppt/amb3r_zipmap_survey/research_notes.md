# AMB3R 与 ZipMap 调研笔记

更新日期：2026-04-27

## 资料入口

- AMB3R 论文：https://arxiv.org/abs/2511.20343
- AMB3R 项目页：https://hengyiwang.github.io/projects/amber
- AMB3R 代码：https://github.com/HengyiWang/amb3r
- ZipMap 论文：https://arxiv.org/abs/2603.04385
- ZipMap 项目页：https://haian-jin.github.io/ZipMap/
- ZipMap 代码：https://github.com/Haian-Jin/ZipMap

## AMB3R：核心判断

AMB3R 的主张是：前馈式 pointmap 模型虽然已经能统一相机、深度、位姿和重建，但它们通常在 2D token/2D grid 上推理，缺少显式 3D 空间紧凑性。真实多视角重建里，多个像素常对应同一个 3D 点；传统 voxel、TSDF、SDF、NeRF 等表示天然要求同一 3D 坐标只有一个几何/外观属性，这会迫使多视角观测融合。AMB3R 把这种空间表示重新引入 pointmap foundation model。

## AMB3R：方法拆解

1. 前端采用 VGGT，负责预测 pointmap、相机、深度和几何特征。
2. VGGT 前端被冻结，避免破坏已有的 attention 和 confidence 函数。
3. 额外训练 metric-scale head，从冻结特征中恢复真实尺度。
4. 后端把 pointmap 与几何特征投到稀疏体素网格；同一 voxel 内的多视角特征被融合。
5. 稀疏 voxel 通过 space-filling curve 序列化为 1D token。
6. 使用 Point Transformer v3 风格的 U-Net 结构做 3D 空间推理。
7. 处理后的 voxel 特征经 KNN interpolation 回到每个像素/点。
8. 用 zero-convolution 注入 VGGT decoder，复用预训练前端能力。

关键细节：

- voxel size 设置为 normalized space 下 0.01，因此真实分辨率会随场景尺度自适应。
- metric scale 不直接回归全局尺度，而是回归每帧“预测深度中位数像素”的 metric log depth；推理时取多帧尺度中位数。
- 训练后端约 40 epochs，每 epoch 2000 samples，总计约 80K samples；论文正文称后端训练约 50 H100 GPU hours，讨论部分按整体 add-on 成本约 80 H100 hours 估算。
- 训练数据包含 ScanNet、ScanNet++、WildRGBD、Mapfree、Aria、Waymo、Virtual KITTI2、GTA-SfM、MVS-Synth、OmniObject3D、Hypersim 等多源数据。

## AMB3R：VO/SfM 扩展

AMB3R 只按多视角重建训练，但论文声称可无任务微调、无 test-time optimization 地扩展到 uncalibrated VO 和 large-scale SfM。

VO：

- 维护 active keyframe memory 和 global keyframe memory。
- 新帧与 keyframes 一起输入模型，预测局部 pointmap、pose、confidence。
- 利用 pointmap 模型“输出在参考帧坐标系中、只差未知尺度”的先验，避免显式 Kabsch-Umeyama 刚体对齐。
- 通过 keyframe 几何估计相对尺度，再用多个 keyframe 的相对位姿加权平均把局部结果对齐到全局。
- active keyframes 有容量上限，补充/重采样 keyframes 后继续在线建图。
- TUM runtime 报告：RTX 4090，392x518 输入，约 4.2 FPS，最好 6.0 FPS，最差 3.4 FPS；active keyframes 上限 10，因此在线复杂度不随总帧数增长。

SfM：

- 使用 image clustering 把大图像集拆成小簇。
- coarse registration 选择高置信簇初始化地图，并逐步把未注册簇映射到全局 keyframes。
- global mapping 两阶段细化：先细化 keyframes，再细化 non-keyframes。
- 仍不做传统 BA，因此它是 feed-forward SfM，而不是优化式 SfM。

## AMB3R：结果与边界

论文评估 7 类任务、13 个数据集：camera pose、monocular depth、multi-view metric depth、3D reconstruction、video depth、VO/SLAM、SfM。

代表性结论：

- Re10K camera pose AUC@30：AMB3R 报告 86.3，高于表中 VGGT run 和多种 pointmap baseline。
- RMVDB multi-view depth：AMB3R 平均 rel 报告 1.7，优于 VGGT、pi^3、MapAnything 等。
- 3D reconstruction：ETH3D、DTU、7-Scenes 上整体优于或持平强基线，DTU 物体级可达到毫米级。
- TUM VO：AMB3R KF 平均 ATE RMSE 2.7 cm；TUM SLAM keyframe 版本也为 2.7 cm。
- ETH3D SLAM VO：AMB3R KF 平均 2.0 cm。
- TUM Dynamic：AMB3R 平均 1.9，和动态专用/带后处理方法接近。
- ETH3D SfM：平均 RRA@5/RTA@5 为 98.2/81.9，显著高于表中 COLMAP、VGGSfM、DF-SfM、MASt3R-SfM。
- Tanks&Temples SfM：Training/Intermediate/Advanced 三档分别 95.0/94.5、98.7/96.9、68.0/72.4。

局限：

- 后端仍是局部场景表示，不能跨所有 chunks 做真正全局联合推理。
- 基础 type-c/VGGT 类模型仍存在输入图像数二次复杂度问题。
- 未针对动态场景训练，动态物体占主导时可能失败。
- VO 依赖 dense reconstruction prior；深度范围复杂、细结构多、远景主导时尺度对齐会不可靠。
- 没有显式 loop closure / relocalization，长期大场景或 kidnapping 场景会漂移/失败。
- SfM 初始化依赖图像聚类和 confidence，可能把视觉相似但无几何重叠的图像分到一起并产生幻觉几何。
- feed-forward poses 不一定严格满足几何约束；NVS 等下游任务可能仍需要 BA 后处理。

## ZipMap：核心判断

ZipMap 解决的是另一类瓶颈：VGGT、pi^3 等强前馈 3D 模型依赖全局 attention，输入图像数 N 增长时成本为 O(N^2)，很难处理几百张图像。顺序模型能做到 O(N)，但常牺牲重建质量并容易累积误差。ZipMap 用 Test-Time Training layers 把整组图像压缩到紧凑的 hidden scene state，从而实现线性时间、双向、单次前馈重建。

## ZipMap：方法拆解

1. 每张图像先由 DINOv2 encoder 得到 patch tokens。
2. 每张图像配一个 camera token 和 register tokens；query 输入使用 query token。
3. backbone 由 24 个 block 组成，交替使用 per-frame local window attention 和 large-chunk TTT layer。
4. TTT layer 不维护随 N 增长的全局 token buffer，而是把上下文写入 MLP fast weights。
5. 这些 fast weights 既是全局信息通道，也是隐式 scene state。
6. 模型一次前馈输出 camera poses、depth maps、point maps。
7. 同一个 state 可在 novel camera/ray map 条件下被实时查询，输出 RGB/depth/colored point map。

实现细节：

- patch size 14，token dimension d=1024。
- fast-weight MLP intermediate dimension 2048，state size 约 6d^2 per layer。
- 损失包括 point loss、depth loss、camera loss，以及 query finetune 阶段的 RGB/depth query loss。
- 训练分三阶段：静态数据带 reference view 80K iterations，动态数据 fine-tune 40K iterations，移除 reference view 后再训 60K iterations。
- 训练使用 64 H100 GPUs；训练数据来自 29 个公开数据集，其中 23 个静态数据集和 6 个动态数据集。

## ZipMap：能力点

长序列重建：

- 项目页与论文摘要称可在单张 H100 上重建 700+ frames，耗时低于 10 秒。
- runtime appendix 说明 750 frames 下低于 10 秒，约 75 FPS，相比 VGGT 快 20x+，相比 pi^3 快 15x+。
- 在 5 帧极短输入时，ZipMap 可能比高度优化的 FlashAttention 全局 attention 基线略慢；优势主要在 N 增大后出现。

状态查询：

- 查询 scene state 时只执行 TTT block 的 apply 操作，不做 update，因此约 100 FPS。
- 可在新视角 query colored point map、RGB 和 depth。
- 对未观察区域能推断墙、地板、地面等常见结构，但不会可靠生成高频细节或完整未见物体。

流式重建：

- streaming 版本通过逐帧更新 TTT scene state 实现在线重建。
- streaming fine-tune 使用 32 H100 GPUs，先 12-view context 训练 60K steps，再 24-view context 训练 30K steps。
- 作者指出若训练上下文进一步增至 64 views，可能继续提升。

## ZipMap：结果与边界

评估覆盖 camera pose、point map、video depth、monocular depth、long-sequence runtime、state query、streaming。

代表性结论：

- RealEstate10K / Co3Dv2 camera pose：ZipMap 在 O(N) 方法中表现强，RealEstate10K AUC@30 高于 CUT3R、TTT3R 等线性方法。
- Sintel / TUM Dynamics / ScanNet camera pose：总体接近或优于 VGGT，并与 pi^3 处在同一强基线水平。
- 7-Scenes / NRGBD point map：dense/sparse 设置下接近或超过 VGGT，明显优于 CUT3R、TTT3R 等线性方法。
- Video depth：Sintel、Bonn、KITTI 上优于多数 O(N) baselines，并通常超过 VGGT。
- Long sequence：ScanNet-v2、DL3DV、7-Scenes 上随着 N 增大，ZipMap 的误差增长明显小于 CUT3R/TTT3R，并接近 VGGT/pi^3。

局限：

- 超长序列且场景规模远超训练分布时仍会退化，这是现有 feed-forward 方法共同问题。
- query RGB 不是 NVS 主任务；高频区域会模糊，不声称达到 unposed novel view synthesis SOTA。
- TTT block 当前用标准 PyTorch 和 Newton-Schulz 正交化，短序列常数开销较大。
- 查询状态的几何/外观是隐式压缩结果，不等价于可编辑、可显式约束的 SLAM map。

## 展示页讲解策略

不要把两篇文章过度绑定。建议讲解顺序：

1. 先建立同一背景：前馈 3D 模型已经从 pairwise pointmap 发展到多视角/长序列，但仍有空间推理、尺度、复杂度三个瓶颈。
2. AMB3R 单独讲：它解决空间紧凑性和 metric-scale/VO/SfM 任务泛化。
3. ZipMap 单独讲：它解决 O(N^2) 可扩展性，并把 hidden state 变成可查询场景状态。
4. 最后只给一页简短对照：AMB3R 像“几何后端增强”，ZipMap 像“序列状态压缩”，不强行说谁替代谁。

