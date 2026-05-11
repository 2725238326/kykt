#import "simple-typst-thesis/template.typ": project

#show: project.with(
  title: "近期 3R 模型及其应用路径综述",
  authors: (
    (
      name: "KYKT Dream",
      email: "",
      affiliation: "Internal research draft",
      postal: "",
      phone: "",
    ),
  ),
  abstract: [
    近年来，基于图像的三维重建逐渐从显式匹配、相机位姿估计、三角化和多视图融合流程，转向以可学习几何表示为核心的 feed-forward 3D reconstruction 路线。DUSt3R 以 pointmap 形式把深度、位姿、匹配和稠密重建纳入同一几何输出，随后 MASt3R、Fast3R、CUT3R、Spann3R、MonST3R、VGGT 等工作分别沿匹配增强、多视角规模化、长序列状态、空间记忆、动态场景和统一视觉几何预测展开。本文围绕近期 3R 模型谱系进行综述，重点比较输入假设、输出表示、相机依赖、动态和长序列能力、测试时验证机制以及可视化输出路径。文中区分原论文结论、代码或 demo 状态、本地流程验证和应用可行性判断；对尚未直接验证的性能、SOTA、许可证和工程复现信息，保留“尚需确认”的标注。
  ],
)

= 引言：3R 方法的范式变化和综述范围

传统基于图像的三维重建通常由若干相对独立的阶段组成：先提取局部特征并建立匹配，再估计相机内外参或相对位姿，随后通过三角化、多视图深度估计和全局优化得到点云、网格或可渲染表示。这一路线具有清晰的几何解释，也便于在 SfM、MVS 和 SLAM 系统中逐步调试；但它对纹理、视角覆盖、相机标定、外点剔除和全局优化质量较敏感。在低纹理、宽基线、稀疏视角、动态物体或长视频场景中，单个阶段的误差往往会沿流程累积，最终表现为漂移、尺度不稳、匹配断裂或局部几何污染。

近期 3R / feed-forward 3D reconstruction 方法的变化不只是把传统流程中的某一步替换为神经网络，而是重新定义了中间表示。DUSt3R 将图像对直接映射为 dense pointmap 和 confidence，并以此导出深度、匹配、位姿和重建结果 @dust3r。后续工作没有沿单一路线发展，而是在不同应用条件下分化：MASt3R 强调 3D-grounded matching @mast3r，Fast3R 面向多图规模化 @fast3r，CUT3R 和一批后续工作处理长序列状态 @cut3r，MonST3R 等方法处理动态视频 @monst3r，VGGT 则把 camera、depth、pointmap 和 tracks 放进统一的视觉几何预测框架 @vggt。

本文的目标不是给出一个简单的排行榜。不同模型面对的输入 regime 和输出需求并不相同：图像对匹配、稀疏多视角、千图级 batch、在线视频流、动态场景、可视化 Gaussian 输出，各自的约束差别很大。因而本文把模型组织为若干能力线索：pointmap 与 matching，多视角规模化，视频深度与动态场景，长序列状态和记忆，测试时验证与自适应，以及面向应用的输出表示。

本文使用 Dream 长期调研中的经验作为方法论背景，而不把 Dream3R 写成综述中心。这里借鉴的主要是证据纪律：论文声称、代码可用、demo 可运行、本地流程跑通、真实质量领先，是五类不同判断。若没有直接来源或实验验证，本文不把它们合并成一个结论。

= 从传统几何流程到 pointmap 表示

SfM/MVS 流程的优势在于每个中间变量都有明确几何含义。特征匹配给出跨图像对应，基础矩阵或本质矩阵约束相对相机位姿，三角化把匹配转为三维点，bundle adjustment 再统一优化相机和点的位置。这个结构适合可解释调试，也使得工程系统可以在局部失败时替换某个模块。然而，流程越长，越依赖前面阶段的质量；当匹配稀疏或相机估计不稳时，后续融合和优化很难凭空恢复缺失几何。

DUSt3R 式 pointmap 表示改变了这一组织方式。模型不先要求完整相机标定或位姿，而是从图像内容直接预测每个像素对应的三维点，并同时给出 confidence @dust3r。这样，深度图、稠密匹配、相对相机和点云都可以从同一组预测中派生出来。Pointmap 并不取消几何约束，而是把部分原先显式求解的中间变量收进可学习表示，再通过 alignment、matching 或下游优化取回。

这种变化带来三个直接影响。第一，模型更适合处理相机信息缺失或不可靠的输入，尤其是互联网图片、稀疏视角和无标定图像。第二，输出天然包含 confidence，便于后续做质量筛选、失败区域标记和测试时一致性检查。第三，pointmap 也为动态和长序列工作提供了统一接口：后续模型可以围绕点图、状态、缓存或时间一致性扩展，而不必完全重写传统几何流程。

但 pointmap 不是终点。DUSt3R 之后的大量工作恰恰说明，单次前向预测仍无法自然解决所有问题。多视角时需要全局一致性，长序列时需要状态和记忆，动态场景中静态几何假设会被运动物体破坏，应用端还需要把点云、深度或 Gaussian 转为可查看、可报告、可复现的产物。这些问题构成了近期 3R 模型继续分化的主要动因。

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 7pt,
    align: left,
    [传统显式流程], [Learned pointmap 路线],
    [feature extraction -> matching -> pose estimation -> triangulation -> MVS/depth fusion -> point cloud / mesh],
    [image pair/set -> transformer prediction -> pointmap + confidence -> alignment / matching / pose / depth / output],
  ),
  caption: [传统重建流程与 pointmap 路线的结构差异。右侧并不消除几何，只是把部分中间变量纳入可学习的稠密 3D 表示。],
)

= DUSt3R 及匹配增强分支

== DUSt3R：pose-free dense pointmap reconstruction

DUSt3R 的关键贡献是把“几何 3D 视觉”转化为直接预测稠密三维点图的问题 @dust3r。在它之前，深度估计、相机位姿、图像匹配和三维重建常常被拆成不同任务；DUSt3R 把这些任务连接到一个统一输出上，使得无相机先验的图像对也可以获得可用于全局对齐的三维表示。对综述而言，DUSt3R 的意义在于提供了后续模型共同继承的接口，而不仅是一种具体网络结构。

DUSt3R 的局限也同样重要。它并不自动解决长序列漂移，也不天然处理复杂动态物体；在多视角输入下，如何高效组织成对预测和全局对齐仍是系统问题。因此后续方法大多不是简单“改进 DUSt3R”，而是在不同输入条件下处理特定瓶颈。

== MASt3R：3D-grounded matching 与 descriptor grounding

MASt3R 的目标是把图像匹配建立在 3D-grounded 表示之上 @mast3r。相比只把 pointmap 看作重建结果，MASt3R 更强调 dense local features、reciprocal matching 和 sparse global alignment，使其更自然地接入匹配、检索和 SfM 场景。它和 DUSt3R 的关系不宜写成“替代”，而应理解为同一几何表示向 matching 任务的扩展。

MASt3R-SfM 进一步把 MASt3R 特征、检索和全局 SfM alignment 组织为完整无约束 SfM 方案 @mast3r_sfm。它对本文的意义有两层：一方面，它说明 learned 3D features 可以反过来强化传统 SfM；另一方面，它也提醒我们 feed-forward 3R 和经典几何后处理并非对立关系，许多真实系统会同时使用二者。

== 基础分支的小结

DUSt3R、MASt3R 和 MASt3R-SfM 共同奠定了本文的基础线索：pointmap 提供统一几何输出，matching 把这种输出变成跨图像 correspondence，SfM alignment 则把 learned matching 接回可解释的全局几何流程。后续章节中的多视角、长序列、动态场景和 Gaussian 输出，基本都可以看作围绕这一接口继续扩展。

= 多视角规模化与统一视觉几何模型

多视角场景首先遇到的是计算和组织问题。如果仍以图像对为基本单位，图像数量增加后，成对组合、全局对齐和内存成本会迅速上升。Fast3R 明确把问题设定为许多图像的一次前向重建，论文题目即强调 1000+ images in one forward pass @fast3r。这里的“Fast”不应被简化为速度宣传；更准确的理解是它面向 many-view regime，试图减少 pairwise pipeline 和全局对齐带来的系统负担。

MV-DUSt3R+ 处理的是稀疏多视角 pose-free reconstruction，它以 single-stage scene reconstruction from sparse views 为目标 @mvdust3rplus。与 Fast3R 的 many-view 侧重点不同，MV-DUSt3R+ 更贴近 sparse-view 输入和可视化输出需求，适合放在多视角重建与应用产物之间讨论。代码、权重和 demo 状态可以作为工程可复现性的线索，但不应直接写成质量结论。

VGGT 是近期统一 visual geometry prediction 的代表。它把 camera parameters、depth maps、point maps 和 point tracks 放进一个 feed-forward Transformer 体系 @vggt。这一路线的重要性在于，它不再只解决“从图像到点云”的单一输出，而是把多种视觉几何变量组织为同一模型的预测目标。与此同时，VGGT 也不应被写成“一步替代整个 3R 家族”。它的优势和限制需要按输入规模、场景类型、显存成本、是否需要在线处理等条件分别讨论。

MapAnything 进一步把问题推进到 universal feed-forward metric 3D reconstruction，允许一张或多张图以及可选的内参、位姿、深度或 partial reconstruction 等输入条件 @mapanything。它体现了另一种趋势：模型不再假设单一输入格式，而是把不同先验统一纳入 feed-forward metric reconstruction。这个方向对应用很有吸引力，但综述写作必须避免把“输入形式更通用”推导成“所有场景都更可靠”。具体性能仍应回到论文实验或后续评测。

Pow3R 则更直接地讨论 camera 和 scene priors 如何增强无约束重建 @pow3r。它提示一个实际问题：许多应用并非完全无先验，相机内参、稀疏深度、已有场景片段或粗位姿都可能存在。合理使用这些先验，可以降低问题难度；但先验错误时也会引入冲突。因此，先验利用和先验验证应同时进入系统设计。

= 视频深度、动态场景和 4D 重建

视频输入给 3R 带来两个相互纠缠的问题：一是帧间几何需要时间一致，二是运动物体会破坏静态场景假设。Align3R 连接了单目深度估计与 3R 对齐，它将 monocular depth 在动态视频中对齐，并利用 DUSt3R 式几何关系改善时序一致性 @align3r。Depth Anything、Depth Anything V2、Depth Pro 和 Metric3D v2 等模型提供了强单帧深度先验 @depth_anything @depth_anything_v2 @depth_pro @metric3dv2，但单帧深度质量不能直接等同于跨帧三维一致性。

MonST3R 是动态场景中最直接的 DUSt3R 后继之一。它把 geometry estimation 放到 presence of motion 的条件下处理 @monst3r，提供动态视频中的几何输出、dynamic masks 和 confidence 线索。需要注意的是，MonST3R 的输出可以作为动态区域分析的输入，但它本身并不等于长期 object identity memory；这一点在应用系统中尤其容易混淆。

POMATO、D^2USt3R、Easi3R 和 RayMap3R 进一步说明动态 3R 尚未收敛为单一机制。POMATO 将 pointmap matching 与 temporal motion 结合 @pomato；D^2USt3R 以 4D pointmaps 处理动态场景 @d2ust3r；Easi3R 从 DUSt3R 出发做 training-free motion disentanglement @easi3r；RayMap3R 则使用 inference-time RayMap 来区分静态和动态结构 @raymap3r。它们共同面对的是静态几何被运动破坏的问题，但采用的表示和代价不同。

动态和 4D 章节还需要与 4DGS 输出区分开。D^2USt3R 的“4D”指随时间变化的 pointmap 表示；4D Gaussian Splatting 则是面向动态新视角合成和渲染的表示 @gaussian_splatting_4d @rotor_4dgs。前者更接近几何预测，后者更接近可视化资产。二者可以在应用链路上衔接，但不应在概念上混用。

= 长序列重建中的状态、记忆和缓存机制

长序列 3R 的核心问题是时间跨度。短视频或图像集合可以依赖一次前向预测或较小窗口；长视频则需要决定哪些历史信息保留、如何更新状态、如何控制缓存预算以及如何避免动态物体污染静态地图。近期模型在这里分化出多个概念：recurrent latent state、spatial memory、pointer memory、KV/cache、hybrid memory、pose-adaptive update 和 latent filtering。

CUT3R 以 persistent state 处理 continuous 3D perception @cut3r。它的状态是 recurrent latent state，不是可任意写入、按空间查询的外部数据库。Spann3R 使用 spatial memory 进行 3D reconstruction @spann3r，更接近显式空间记忆；Point3R 则进一步强调 explicit spatial pointer memory @point3r。把这三者都称作“记忆增强”会遮蔽关键差异：存储对象、查询方式和更新规则并不相同。

STream3R 和 LongStream 代表 causal / autoregressive streaming route。STream3R 将 sequential 3D reconstruction 放进 causal Transformer 框架 @stream3r，LongStream 则讨论 long-sequence streaming autoregressive visual geometry @longstream。它们与 CUT3R 的 persistent state 不完全相同，更强调流式会话、因果处理和缓存刷新。

LONG3R、LoGeR 和 Mem3R 则集中在 long-context memory 上。LONG3R 处理 long sequence streaming reconstruction @long3r；LoGeR 使用 hybrid memory 进行 long-context geometric reconstruction @loger；Mem3R 则通过 test-time training 组织 streaming 3D reconstruction with hybrid memory，并强调 tracking 与 mapping 的解耦 @mem3r。这些方法都说明长序列 3R 已经从“能否处理更多帧”转向“如何治理历史信息”。

OVGGT、PAS3R 和 FILT3R 提供了更细的状态治理视角。OVGGT 关注 O(1) constant-cost streaming visual geometry Transformer @ovggt，核心是固定预算下的 cache compression 和 anchor protection；PAS3R 用 pose-adaptive update 处理长视频序列 @pas3r；FILT3R 则把 Kalman-style filtering 引入 latent state @filt3r。它们提示，长序列 3R 的关键不只是存储更多，而是选择何时写入、何时保留、何时忽略和何时重置。

#figure(
  table(
    columns: (1.2fr, 1.8fr, 2.2fr),
    inset: 6pt,
    align: left,
    [机制], [代表方法], [写作边界],
    [Recurrent state], [CUT3R, STream3R], [压缩状态随输入更新，不等同于外部空间数据库。],
    [Spatial / pointer memory], [Spann3R, Point3R], [按空间或指针组织历史几何，需要讨论写入和查询规则。],
    [Hybrid memory], [LONG3R, LoGeR, Mem3R], [本地/全局、tracking/mapping 等轴线不同，不能简单合并。],
    [Cache and update governance], [OVGGT, PAS3R, FILT3R, LongStream], [关注预算、更新增益、滤波和缓存刷新。],
  ),
  caption: [长序列 3R 中常被混写的四类机制。],
)

= 测试时验证、修正和自适应

当 3R 模型进入真实应用时，错误检测和修正机制往往比单次重建指标更接近系统需求。Test3R、TTT3R、G-CUT3R 和 MASt3R-SfM 从不同角度触及这个问题，但它们的作用范围不同。

Test3R 的名称容易造成误解。按 Dream 本地文献板的区分，它侧重 test-time geometric consistency，对 DUSt3R/MASt3R family 输出进行一致性评分和重建质量相关性建模 @test3r；它不是对模型参数或状态进行训练式更新。TTT3R 则明确把 3D reconstruction 作为 test-time training 问题处理 @ttt3r。二者都发生在测试阶段，但一个偏验证和评分，一个偏状态/模型更新，计算代价和失败模式不同。

G-CUT3R 讨论 camera 和 depth priors 的 guided reconstruction @gcut3r。它适合放在先验使用和测试时修正之间：当有相机、深度或校准信息时，模型可以受先验约束；但先验本身也可能错误或与 RGB evidence 冲突。本文只把“先验冲突检测”作为系统设计层面的推论，不写成 G-CUT3R 论文已经直接解决的问题。

MASt3R-SfM 提供了另一种传统几何式一致性路径 @mast3r_sfm。它通过 matching、retrieval 和 global SfM alignment 组织结果，适合处理图像集合和 SfM-stage refinement。与 Test3R 的 per-window consistency 相比，MASt3R-SfM 更重、也更贴近传统全局几何系统。真实应用可以同时需要二者：轻量一致性检查用于早期发现错误，SfM/global alignment 用于更完整的几何整理。

= 面向应用的输出表示

3R 模型的论文输出不等于应用产物。应用端通常需要可查看、可比较、可报告的结果：深度图、点云、confidence map、失败区域、相机轨迹、mesh、Gaussian 或新视角渲染。Pointmap 是良好的中间表示，但用户和报告系统往往不会直接阅读 pointmap；它需要被转化为图、表、三维 viewer、对比截图或质量日志。

3D Gaussian Splatting 提供了实时可渲染表示 @gaussian_splatting_3d，4DGS 扩展到动态场景 @gaussian_splatting_4d @rotor_4dgs。在 3R 综述中，它们应作为输出表示和可视化层出现，而不是被写成 pose-free 3R 的同义词。原始 3DGS 通常依赖相机位姿或 SfM 初始化；pose-free Gaussian 方法的意义在于把前端几何预测与 Gaussian 输出进一步靠近。

Splatt3R、InstantSplat 和 NoPoSplat 分别提供了从少量或未标定图像到 Gaussian 表示的路线。Splatt3R 处理 uncalibrated image pairs 到 zero-shot Gaussian splatting @splatt3r；InstantSplat 强调 sparse-view Gaussian Splatting in seconds @instantsplat；NoPoSplat 处理 sparse unposed images 到 3D Gaussian splats @noposplat。这一分支对演示和应用很重要，因为它能把几何预测转为更容易观察的资产。但可视化吸引力不等于几何质量领先，许可证、依赖、权重和本地复现状态也必须分开记录。

应用路径可以概括为：输入图像或视频首先被识别为图像对、稀疏多视角、many-view、streaming 或 dynamic video；模型输出 pointmap、depth、camera、tracks 或 memory state；随后根据 confidence、一致性残差、动态区域和先验冲突等证据判断是否接受、修正或换模型；最终再转为点云、mesh、Gaussian、截图或报告。KYKT 类系统可以承担结果记录、失败样本归档和证据日志，但不应把系统流程跑通写成模型质量结论。

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 6pt,
    align: center,
    [输入], [几何预测], [质量证据], [输出表示], [报告/集成],
    [图像对、多视角、视频], [pointmap、depth、camera、tracks], [confidence、一致性、动态区域、先验冲突], [点云、mesh、3DGS、4DGS], [截图、表格、失败区域、证据日志],
  ),
  caption: [从 3R 模型到可用产物的应用路径。质量证据用于辅助判断，不自动构成质量领先结论。],
)

= 方法比较表

#table(
  columns: (1fr, 1.2fr, 1.4fr, 1.2fr, 1.3fr, 1.5fr),
  inset: 5pt,
  align: left,
  [模型], [输入假设], [主要输出], [相机/位姿], [时间/动态], [应用备注],
  [DUSt3R], [图像对/多视角], [pointmap, confidence], [pose-free], [静态为主], [基础线，代码和 demo 已列入 registry。],
  [MASt3R], [图像对/集合], [3D-grounded matching], [pose-free], [静态匹配], [适合 matching 和 SfM 接口。],
  [Fast3R], [many-view], [多视角重建], [弱化 pose 输入], [非流式], [规模化 baseline；本地依赖状态需保守记录。],
  [VGGT], [多视角], [camera/depth/pointmap/tracks], [统一预测], [非专门动态], [强统一模型；具体 regime 需实验比较。],
  [CUT3R], [连续视频], [pointmap/state], [pose-free/弱先验], [persistent state], [长序列状态 baseline。],
  [Spann3R], [序列/多视角], [global pointmap], [pose-free], [spatial memory], [与 Point3R/Mem3R 机制需区分。],
  [MonST3R], [动态视频], [geometry/masks/confidence], [pose-free], [动态场景], [动态 3R baseline，不等于 object memory。],
  [Test3R], [3R 输出/图像 triplet], [consistency signal], [依赖 3R 输出], [测试时验证], [不是 test-time training。],
  [TTT3R], [CUT3R-style cases], [updated state], [依赖基础模型], [测试时更新], [计算和稳定性需单独评估。],
  [Splatt3R/InstantSplat/NoPoSplat], [未标定或稀疏图像], [3D Gaussians], [pose-free/sparse], [静态或短序列为主], [可视化输出强，质量和许可需分开确认。],
)

这个表只给出综述中的第一层分类。最终若要写性能或 SOTA 结论，需要回到每篇论文的实验设置：数据集、输入视角数、是否使用相机、是否包含动态场景、是否做全局优化、是否允许测试时更新、是否比较相同输出表示。缺少这些条件时，跨论文比较很容易变成不成立的横向排名。

= 图示计划

本文最终建议保留五类图示。第一，3R 模型谱系图，从 DUSt3R 出发，分出 matching、多视角、长序列、动态、测试时验证和 Gaussian 输出几条线。第二，能力分类图，用输入 regime、输出表示、相机依赖、时间处理和应用状态组织模型。第三，传统流程与 pointmap pipeline 对照图，用于解释范式变化。第四，长序列 memory primitives 图，用于避免把 state、memory、cache 和 filtering 混写。第五，应用落地图，展示从图像/视频输入到几何预测、质量证据、输出表示和报告集成的路径。

当前图示优先使用 Typst 原生表格和流程草图。若后续使用 AI 生成图，应使用 `notes/figure_prompts.md` 中记录的提示词，并在图注中明确该图是概念重绘，而不是某篇论文原图。

= 经验总结与开放问题

从 Dream 调研经验看，3R 方向最容易出现两类过度声称。第一类是把模型流程跑通写成质量领先；第二类是把某个模型的局部能力扩展成全场景适用。前者需要用数据集、指标、失败样本和可复现脚本约束，后者需要按输入 regime 拆开：图像对、稀疏视角、many-view、streaming、dynamic video 和 asset output 不是同一问题。

开放问题首先是长序列稳定性。当前方法已经提出 recurrent state、spatial memory、hybrid memory、cache compression 和 filtering，但仍需要更清晰的评测来回答：哪些历史信息应该保留，哪些应当被遗忘，动态物体如何避免污染静态地图，cache budget 与几何质量之间如何权衡。其次是动态和 4D 表示。MonST3R、POMATO、D^2USt3R、Easi3R 和 RayMap3R 给出了不同方向，但 object identity、static/dynamic separation 和可渲染动态资产之间仍存在接口断裂。

第三个问题是测试时机制。Test3R、TTT3R 和 G-CUT3R 表明测试阶段可以做一致性检查、状态更新或先验引导，但它们引入的额外计算、更新风险和先验冲突需要严格记录。第四个问题是应用输出。点云、深度图、confidence map、mesh 和 Gaussian 各有用处；对教学、报告、工程验收或系统集成来说，能否稳定产出可查看证据，常常比单张 demo 是否好看更重要。

因此，后续工作不应只追求“一个模型覆盖所有场景”。更稳妥的路线是建立模型能力卡和样本 regime 卡：先判断输入条件，再选择合适模型或组合，最后用几何证据和可视化产物记录结果。这样的框架不替代论文实验，但能降低应用端把不同模型误用到不匹配场景的风险。

= 结论

DUSt3R 之后的 3R 模型谱系已经从单一 pointmap 重建扩展为一个多分支生态：MASt3R 和 MASt3R-SfM 强化匹配与传统 SfM 接口，Fast3R、MV-DUSt3R+、VGGT 和 MapAnything 推进多视角与统一视觉几何，CUT3R、Spann3R、Point3R、STream3R、LONG3R、LoGeR、Mem3R、OVGGT 等工作处理长序列状态和记忆，MonST3R、POMATO、D^2USt3R、Easi3R、RayMap3R 面向动态与 4D 场景，Test3R、TTT3R、G-CUT3R 则把测试时验证、更新和先验引导纳入讨论。Gaussian 相关方法把几何结果推向可查看资产，但也带来许可证、依赖和质量验证问题。

这一领域的关键不在于寻找一个可以覆盖所有输入的单一答案，而在于理解不同方法的输入假设、输出表示和失败边界。对应用系统而言，可信的 3R 结果需要同时包含几何预测、质量证据、可视化输出和复现记录。性能排名、SOTA 判断和部署建议只有在实验设置、代码状态与许可证条件同时明确时才具有实际意义；在这些条件缺失时，保守的证据标注比概括性结论更可靠。

#pagebreak(weak: true)
#set page(header: [])
= Bibliography
#bibliography("bib.yaml", style: "apa", title: none)
