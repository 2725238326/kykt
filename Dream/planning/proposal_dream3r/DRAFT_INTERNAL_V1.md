# Dream3R 开题报告（内部稿 V1）

| 字段 | 取值 |
|---|---|
| 文件类型 | 开题报告内部稿 (Dream-vocabulary; 不外发) |
| 创建日期 | 2026-05-16 |
| 最后更新 | 2026-05-16 (cycle 037 §2 ~4200 字 + cycle 038 §4 ~4000 字 + cycle 039 §3 ~1800 字 + §6 ~1300 字, 累计正文 +~11300 字) |
| 状态 | v1 draft; §1 + §2 + §3 + §4 + §6 完整起草; §5 / §7 / §8 / §9 placeholder, 待 cycle 040+ 起草 |
| 授权根 | DEC-20260516-001 (cycle 036 launch) + DEC-20260516-002 (cycle 037 §2) + DEC-20260516-003 (cycle 038 §4) + DEC-20260516-004 (cycle 039 §3 + §6) |
| 配套文件 | OUTLINE_V1.md (章节结构) + STYLE_CONTRACT.md (双稿契约) + DRAFT_EXTERNAL_V1.md (外部稿) |
| 双稿关系 | 本稿是 master per STYLE_CONTRACT §3 规则 1; 外部稿是 internal 周期性快照 |
| 词汇 | 含 Dream / Dream3R / cycle / SPEC / DEC / CR / W-task / Track A / Track B / 服务器 path / agent 等 |

---

## §1 项目背景与研究问题

### 1.1 Track A 主线决策起源

Dream3R 项目 (内部代号 Dream, 架构产品 Dream3R) 的研究方向于 2026-05-06 经 DEC-20260506-001 user-locked 后定型为 architecture-first 主线: 设计新的 3R (前馈式三维重建) 架构作为 markdown spec + ablation plan + comparator map, 把架构本身作为 PRIMARY output; 论文写作降为 SUPPORT artifact。

这一决策的 5 个 in-force constraint 共同约束本开题报告:

- DEC-20260506-001 (architecture-first mainline) — 主线是架构而非论文
- DEC-20260504-002 (no-all-in) — 4 个 finalist 机制 (Critic / Memory / Permanence / Composer) 不收敛到任一单一 finalist
- DEC-20260501-011 (Dream3R thesis reframe; candidate-not-final) — Dream3R 是 candidate 架构, 非最终方案
- DEC-20260503-001 (research-code-discipline) — 5 条纪律 (尤其 rule 3 surgical edits + rule 5 honesty override) 约束本研究的所有 spec / code / cycle 操作
- F-002 (server-side discipline) — KYKT 3R 模型工作在 /hdd3/kykt26/code/dream3r/ 远端服务器执行; 本地 Windows 仅作 markdown + 编排

### 1.2 Dream3R v0.3 当前状态

Dream3R v0.3 主线截至 2026-05-16 处于以下状态:

- **架构定型 (markdown 层)**: SPEC-20260506-004 v0.2 完整定义 6 个核心模块 — C1 Perceiver (DINOv3-S frozen backbone; ViT-L → DINOv3-S 替换 per Delta 2, ~14x 参数减少 + ~5x 延迟加速) + C2 Memory (NSA three-branch 即 compressed/selected/sliding 三分支 + AnchorBank K=256 + StateToken; SPEC-20260508-001 v0.3 addendum) + C3 Permanence (Slot Attention + permanence_link) + C4 Critic (Sampson 几何 / depth 一致性 / 共视 conflict 三类信号 + repair actions 0/1/2 stub 3/4/5) + C5 Composer (7 expert pool: MASt3R + Fast3R + Spann3R + CUT3R + MoGe-2 + DepthAnything-V2 + Test3R; per Delta 5 + COMPOSER_CAPABILITY_DESCRIPTORS) + C6 Bus (CR-1..CR-6 cross-spec signal contract v2.1 per `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md`)
- **比较图谱定型**: SPEC-20260507-001 v0.2 把 19 个 comparator entry 重组为 5 tier (in-pool 7 / out-of-pool 3 / out-of-scope 1 / foundation 1 / orthogonal 8); 引入 Axis 9 NSA / Axis 10 DINOv3 / Axis 11 Composer pool 三个新轴
- **消融计划定型**: SPEC-20260506-005 v0.2 ABL-v02-1..9 + SPEC-20260508-002 ABL-memory-0..11 (后者 cycle 029 review 后 v1.1 修订)
- **代码实装 (服务器层)**: cycle 033 W1-W16 + cycle 034 W17-W18 完成。W1-W18 含 DINOv2 backbone (实际跑) + 3D-aware retrieval + active/stable state + Grassmannian 正则化 + 几何 Critic + ISA slot + 真实 MASt3R + Spann3R adapter + W17 Mamba-Transformer 混合循环 + W18 GaussianHead tensor 契约 (renderer-free)。代码部署在 /hdd3/kykt26/code/dream3r/
- **真实数据集成证据 (非质量证明)**: cycle 034 跑 evaluate_real_sequence.py on KITTI 真实序列 2 windows, pointmap L2 = 20.47。此数值作为系统集成证据 (端到端 pipeline 跑通), 非训练后质量。
- **ablation 现状**: ablate_recurrence.py 实装 4 变体 (baseline_cross_attention / mamba_hybrid / no_nsa / no_stable_memory), 在合成数据 windows=3 跑过。ABL-memory-0 通过 (cycle 031 local P0 scaffold)。其他 ABL 待启动。
- **现有 gated 项**: 真实数据训练 / 3DGS 渲染 / DTU 加载器 / Fast3R omegaconf 依赖 / 真实 CUT3R / MoGe-2 / DepthAnything-V2 / Test3R 加载 / TTT 路径 / W19-W30 等

### 1.3 Track B 综述四轴判断的反哺

Track B 3R-mix 中文综述于 2026-05-14 按 route C (arXiv-only) wound down, 18 A4 页 / 44 引文 / 6 图 5 表 / 0 编译错误。综述识别了 3R 研究方向的四轴判断:

- **轴 A: 六类典型几何失败模式** (综述 §10 首段) — 弱纹理 / 镜面玻璃 / 快速运动 / 长基线 / 尺度漂移 / 域外
- **轴 B: 长序列内存四类机制** (综述 §6) — 空间指针 (Point3R 类) / causal-autoregressive (CUT3R / STream3R 类) / hybrid memory (NSA-hybrid 类) / 预算治理与滤波 (LONG3R 类)
- **轴 C: 测试时机制三类区分** (综述 §7) — 一致性优化 (Test3R 类) / TTT 参数更新 (TTT3R 类) / 先验注入位置区分 (G-CUT3R / Pow3R / MASt3R-SfM 类)
- **轴 D: 输出资产三类** (综述 §8) — 4D pointmap / dynamic mask / 4DGS asset

cycle 035 (2026-05-15) 把这四轴判断映射到 Dream3R v0.3 架构上, 输出 4 个 markdown deliverables (planning/SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL.md + SOTA_MATRIX_V2.md + CRITIC_CALIBRATION_PLAN_V1.md + LONG_SEQ_REAL_TABLE_PLAN.md) + WORK_RISK_REGISTER v1.1 (+4 行: R-OOD-1 / R-EXT-PRIOR-1 / R-4DGS-LIC-1 / R-INPUT-EXT-1)。

综述与 Dream3R 主线关系是单向反哺 (综述 → 主线), 综述 manuscript 在 2026-05-14 wound down 后未受到主线后续工作回流污染 (per Dream/3R-mix/deliverables/RELATION_TO_TRACK_A_2026-05-16.md)。

### 1.4 三个核心研究问题

基于综述四轴判断 + Dream3R v0.3 架构覆盖矩阵 (SOTA_MATRIX_V2 §6 + SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §3 显示 21 子类中 ✓ 6 / ⚠ 11 / ✗ 4), 本研究聚焦三个核心研究问题:

**Q1 验证机制的架构层落地 (Critic 路径; 轴 A + 轴 C)**: 综述 §7 区分 "一致性优化 (Test3R 风格, 无参数更新)" 与 "TTT 参数更新 (TTT3R 风格)" 两类测试时机制。Dream3R v0.3 当前 C4 Critic 是 "验证 + 修复" hybrid, 独立的一致性优化循环未拆出。本研究问: 把验证 (geometric_conflict scoring + repair suggestion) 与适应 (parameter update) 在架构层拆为两条路径, 是否优于 hybrid 配置? (对应 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 B1 v0.4 spec delta 候选)

**Q2 长序列内存四类机制的架构层统一 (Memory 路径; 轴 B)**: 综述 §6 把长序列内存分为 4 类机制, 现有系统每个只占一档 (Point3R = 空间指针; CUT3R = causal-AR; NSA-hybrid 类 = hybrid memory; LONG3R = 预算治理)。Dream3R v0.3 C2 Memory 通过 NSA three-branch + AnchorBank K=256 + StateToken 已覆盖前 3 类 (✓ 空间指针 / ✓ causal-AR / ✓ hybrid memory), 第 4 类 (预算治理) ⚠ partial (帧预算约束接口存在但动态剪枝未对比)。本研究问: 在单一 C2 Memory 架构中同时实装 4 类机制是否可行? 4 类机制间的协同 / 冲突如何在 ablate_recurrence + LONG_SEQ_REAL_TABLE_PLAN V1 KITTI 长窗 (windows ≥ 10) 下显现?

**Q3 多专家组合是否优于单一 expert (Composer 路径; 轴 D + 综述 §3 best-of-N)**: Dream3R v0.3 C5 Composer 7-expert pool (per Delta 5) 是 architecture novelty 之一 (per SPEC-20260507-001 v0.2 pillar D heterogeneous best-of-N)。但综述 §3 + §6 显示这些 expert 各自在不同 regime 上有优势 (MASt3R 静态对 / Fast3R 多视图 / Spann3R 流式 / etc.), 组合是否真的优于单一 expert 是 ABL-v02-4 的 Tier 1 load-bearing 问题。本研究问: 在 KITTI 真实数据上, 多专家 best-of-N 路由是否在 pointmap L2 + route_regret 指标上显著超越单一 expert? 与 Test3R 内置 verifier 组合时, C4 Critic 的额外价值边际如何?

### 1.5 候选 vs 最终的边界

per DEC-20260501-011 (thesis reframe, candidate-not-final) + DEC-20260504-002 (no-all-in), Dream3R 是被评估的候选架构, 非项目收敛方案。本研究的成果不是论证 Dream3R 相对 SOTA 具有压倒性优势, 而是评估 Dream3R 在 Q1 / Q2 / Q3 三组维度上的表现, 为后续 v0.4 spec delta 演进或被替换提供实证依据。

### 1.6 Dream 项目工件引用

本节直接引用的工件:

- `Dream/decisions/DEC-20260506-001-mainline-architecture-first.md` (Track A 主线决策)
- `Dream/decisions/DEC-20260501-011-dream3r-thesis-reframe.md` (candidate-not-final)
- `Dream/decisions/DEC-20260504-002-no-all-in-on-single-finalist.md` (no-all-in)
- `Dream/decisions/DEC-20260503-001-research-code-discipline.md` (5 条纪律)
- `Dream/decisions/DEC-20260515-001-cycle-035-survey-driven-markdown-deliverables-launch.md` (综述反哺主线)
- `Dream/specs/SPEC-20260506-004-dream3r-architecture-v02.md` v0.2 (6 模块 + 6 Delta)
- `Dream/specs/SPEC-20260508-001-dream3r-c2-memory-v03-addendum.md` v0.3 (C2 NSA + AnchorBank)
- `Dream/specs/SPEC-20260507-001-dream3r-comparator-map-v02.md` v0.2 (Tier 1-5 + Axis 9-11)
- `Dream/3R-mix/deliverables/3r_survey_stage_final_2026-05-15_natural.pdf` (Track B 综述 deliverable)
- `Dream/planning/SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL.md` (cycle 035 综述驱动优化提案)
- `Dream/planning/SOTA_MATRIX_V2.md` (综述四轴 + 输入扩展 axis re-label)
- `Dream/code/dream3r/RECENT_PROGRESS.md` (W1-W18 ledger)
- `Dream/cycles/CYCLE-20260511-001.md` (cycle 034 KITTI smoke 实证)

---

## §2 比较谱系与现状

本章在 Track B 3R-mix 中文综述 (Dream/3R-mix/main.tex §2-§9, 18 A4 页, 44 引文) 的基础上重新组织前馈式三维重建 (3R) 的比较谱系, 并在 §2.7 给出综述四轴覆盖矩阵下 Dream3R v0.3 的落点。综述文本本身已 wound down 至 route C arXiv-only 状态 (per cycle 034 + 2026-05-15 prose naturalization deliverable), 本章对其素材作 paraphrase + 结构重组, 不引入综述未触及的新引文。素材锚点逐子节给出。

### 2.1 基础谱系: DUSt3R / MASt3R / MASt3R-SfM 系

DUSt3R (2024) 作为 3R 范式起点提出了 pose-free 稠密点图回归 (per 综述 §3 + tab:foundation 第 1 行), 把图像对映射到稠密三维点图并以置信度反映可靠性, 多视角输入则通过成对预测 + 全局对齐组织。其能力来源于先前 CroCo 跨视角补全自监督预训练所学到的 cross-view 对应能力 (综述 §3 段 1)。新表示的代价是把长序列 / 动态物体 / 遮挡 / 尺度漂移 / 大规模多图等遗留问题推到了下游。

MASt3R (2024) 在 DUSt3R 基础上叠加 dense local feature head, 把描述子直接绑定到三维几何, 使匹配同时受 descriptor 局部判别性与 pointmap 全局几何约束 (综述 §3 段 2)。MASt3R-SfM 进一步把这一匹配能力接回经典 SfM 全局重建。这条支线提示: 3R 表示提供更强先验之后, 传统几何并未被取代而是以新接口回到系统 — 匹配 / 检索 / bundle adjustment 仍在, 输入条件与约束方式发生了变化。

Dream3R v0.3 C5 Composer 7-expert pool (per SPEC-20260506-004 v0.2 Delta 5 + COMPOSER_CAPABILITY_DESCRIPTORS) 显式接受 MASt3R 作为 in-pool 静态对几何专家 (per SPEC-20260507-001 v0.2 Tier 1 in-pool 7); MASt3R-SfM 进入 §2.5 测试时机制讨论。DUSt3R 本身在 SPEC-007 v0.2 中标注为 Tier 4 foundation (基础范式, 不直接进入 expert pool 比较)。

### 2.2 多视角规模化与统一视觉几何

把视角数推上去构成 DUSt3R 类方法的第一类工程压力 (综述 §4 + tab:foundation 后半)。Fast3R 走 many-view one-forward-pass 路线, 单 A100 上扩展至 1500 视角级别前馈重建; MV-DUSt3R+ 在稀疏视角 (12 / 20 视角) 上通过多视角 decoder 与 cross-reference fusion 完成单阶段重建。这两种规模压力方向相反, 都要在"多图汇聚"与"成对对齐"间重新分配计算预算。

VGGT (2025) 把 camera / depth / pointmap / tracks 放在同一视觉几何预测框架下, 把 3R 从"点图重建模型"扩展为"通用几何预测模型" (综述 §4 段 2)。MapAnything 在 metric feed-forward reconstruction 基础上允许内参 / 位姿 / 深度 / 部分重建等多种可选条件作为输入; Pow3R 直接把 camera 与 scene priors 当作可选模态进入前馈预测 (per 综述 §4 + §7)。约束变强了, 条件依赖与先验冲突也跟着进来。

Dream3R v0.3 在比较图谱 (SPEC-20260507-001 v0.2) 中把 Fast3R / MASt3R 列入 Tier 1 in-pool 7 expert; VGGT 列入 Tier 2 out-of-pool dropped (per Delta 5 + DEC-20260507-001 reasons: VGGT offline-batch 与 streaming budget 不兼容); MapAnything 列入 Tier 5 orthogonal (输入扩展 axis 与 v0.3 单图 + 序列输入接口正交)。Pow3R 列入 Tier 5 orthogonal 同时被 cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §3 + SOTA_MATRIX_V2 §6 标注为输入扩展 axis 候选 (R-INPUT-EXT-1 风险源头, 见 §9 风险章节)。

### 2.3 视频、动态场景与 4D 重建

静态场景假设是许多三维重建流程的隐含前提 (综述 §5 + tab:dynamic)。视频输入中一旦出现移动物体 / 非刚体 / 相机快速运动, 单一静态点图会把时间变化错误吸收进几何结构。Align3R 从动态视频单目深度对齐入手 (借 Depth Anything 等深度先验为跨帧一致性提供约束); MonST3R 在 DUSt3R 风格点图上微调, 输出 per-frame geometry + dynamic confidence/mask; POMATO 把 pointmap matching 与 temporal motion 结合; D²USt3R 把动态场景扩展到 4D pointmap; Easi3R 走推理时注意力调整路线, 从既有 DUSt3R 表征中分离运动 (training-free dynamic correction); RayMap3R 利用 RayMap 与图像双分支对比, 借静态偏置识别动态干扰。

注意 4D pointmap 与 dynamic mask 是几何中间量, 4DGS 是可渲染表示 — 两者落到下游做的事并不一样 (此区分在 §2.6 输出资产章节展开)。

Dream3R v0.3 C3 Permanence (Slot Attention + permanence_link, per SPEC-20260506-004 v0.2 §C3) 对应"动静分离"功能, 与 MonST3R / Easi3R 同属一类机制但实现路径不同; SPEC-007 v0.2 把 MonST3R 列入 Tier 5 orthogonal (动态主干路径与 v0.3 静态优先 + Permanence-辅助路径正交)。

### 2.4 长序列重建中的状态、记忆与缓存四类机制

长序列输入把 3R 模型推到第二类常见难题: 在有限算力下保留足够多的历史几何上下文, 又不能让错误状态被一直传下去 (综述 §6 + fig:memory + tab:memory)。综述 §6 把这一方向的近一年工作分为四类互不互斥的机制支线:

**B1 递推状态 (compressed)**: CUT3R 走 persistent recurrent state 的路线, 把连续 3D perception 组织成带状态的模型, 在输入流上递推更新; STream3R 在 causal transformer 框架下做可扩展的逐帧重建; LongStream 通过 gauge-decoupled 的关键帧位姿与正交尺度学习处理长在线序列。

**B2 空间 / 指针记忆 (selected)**: Spann3R 引入外部 spatial memory 用于全局 pointmap 重建; Point3R 在 Spann3R 思路上引入与三维场景结构相关联的指针记忆, 用于流式稠密重建。

**B3 混合记忆 (hybrid)**: LONG3R 用 memory gating 配合 dual-source decoder 维持长序列上下文; LoGeR 在 parametric TTT memory 上叠加滑动窗口注意力; Mem3R 把 tracking 与 mapping 的记忆显式解耦。

**B4 缓存治理与滤波 (budget governance)**: OVGGT 用自选择缓存与动态锚点保护维持固定计算预算; PAS3R 按位姿变化与图像频域线索调整状态更新; FILT3R 在递推状态之上加一层免训练的 Kalman 式潜变量滤波。

Dream3R v0.3 C2 Memory (per SPEC-20260508-001 v0.3 addendum) 通过 NSA three-branch (compressed / selected / sliding) + AnchorBank K=256 + StateToken + Mamba-Transformer 混合循环结构覆盖前 3 类: compressed 分支对应 B1 递推状态 (StateToken 实装); selected 分支 + AnchorBank 对应 B2 空间指针; sliding 分支 + Mamba hybrid 对应 B3 混合记忆。第 4 类 B4 预算治理 ⚠ partial — 帧预算约束接口存在但动态剪枝未对比 (per cycle 035 SOTA_MATRIX_V2 §6.4 + LONG_SEQ_REAL_TABLE_PLAN B4 coverage gap acknowledgement)。这一覆盖关系是 §3 Q2 (长序列内存机制统一) 的核心论据。

### 2.5 测试时验证、修正与先验输入三类机制

3R 模型一般同时输出 pointmap / depth / camera / tracks / confidence 等多个几何量, 它们之间天然存在一致性关系 (综述 §7 + tab:testtime)。综述 §7 把测试时阶段引入的机制分为三类:

**C1 一致性优化 (无参数更新)**: Test3R 用 image triplet 之间的几何一致性做测试时优化, 把推理过程组织为一致性最大化的轻量调整; 不更新模型参数。

**C2 测试时参数更新 (TTT)**: TTT3R 把递推 3R 模型的状态更新当成在线 test-time training, 按对齐置信度推导记忆更新速率; 实质是反向传播在推理阶段动态调整模型参数。

**C3 先验注入 (prior injection)**: G-CUT3R 在 CUT3R 上叠加一组模态特异的先验编码器 (深度 / 相机 / 位姿先验在合适时机进入); Pow3R 把先验当作可选输入直接进入前馈模型, 在训练时就纳入条件建模 (与 G-CUT3R 区别: G-CUT3R 推理时, Pow3R 训练时); MASt3R-SfM 通过经典 SfM 一致性循环对 MASt3R 的匹配结果做校验, 把测试时验证拉回 bundle adjustment 传统范式。综述 §7 段 3 还指出, 除模型自带先验通道外, 3R 系统经常借用外部先验 (Depth Pro / Metric3D v2 / DINOv2 / DINOv3 / CoTracker / SpatialTracker / SAM2), 在尺度 / 匹配 / 动态识别 / 失败检测上提供补充, 但每种先验自己也带偏置和失效区间 — 先验和模型预测对不上时需要在系统层 (而非模型层) 决定听谁的。

Dream3R v0.3 C4 Critic (per SPEC-20260506-004 v0.2 §C4: Sampson 几何 / depth 一致性 / 共视 conflict 三类信号 + repair actions 0/1/2 stub 3/4/5) 是"验证 + 修复"hybrid, 既包含 Test3R 风格的几何一致性验证 (无参数更新), 又包含 repair action 的输出修正 (无参数更新), 但不包含 TTT3R 风格的参数更新路径。这一未拆分的 hybrid 配置正是 §3 Q1 (验证机制路径) 的核心问题: 是否应在架构层把 C1 一致性优化与 C2 TTT 参数更新拆为两条独立路径? 此问题对应 cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 B1 候选 v0.4 spec delta (Critic path split: verification vs test-time adaptation), 仍 proposal-status, 未起草。

### 2.6 从几何预测到可查看输出三类资产

实际应用很少直接面向模型内部的 pointmap (综述 §8)。系统要把几何预测转化成可查看 / 可比较 / 可存档的结果, 这些输出在综述 §8 + tab:application 中归为三类:

**D1 4D pointmap (几何中间量)**: 由 D²USt3R / Dream3R v0.3 C1+C2+C3 输出的稠密时空点图; 适合下游几何一致性检查与跨视角对照, 但不直接渲染。

**D2 dynamic mask (几何中间量, 动静分离)**: MonST3R / Easi3R / RayMap3R / SAM2 输出的动态掩码; 与 D1 配合界定哪些像素属于动态。

**D3 4DGS asset (可渲染输出)**: 3D Gaussian Splatting 提供实时可渲染辐射场表示, 4DGS / 4D-Rotor Gaussian Splatting 把它扩展到动态场景; Splatt3R 在 MASt3R 风格几何上预测高斯属性, 把未标定图像对直接映射到 Gaussian; InstantSplat 借助稠密立体先验与 Gaussian Bundle Adjustment 处理稀疏视角; NoPoSplat 从稀疏无位姿图像直接预测规范坐标下的高斯。

Dream3R v0.3 W17-W18 (per code/dream3r/RECENT_PROGRESS.md, cycle 034 完成) 在 D1 + D2 已有实装 (W17 Mamba-Transformer 混合循环 + W18 GaussianHead tensor 契约 renderer-free); D3 (实际 Gaussian renderer) 仍 gated (W27 candidate per code/dream3r/NEXT_PHASE_ROADMAP.md), 受 R-4DGS-LIC-1 风险约束 (4DGS asset 渲染 license 链未文档化, 见 §9)。

### 2.7 综述四轴覆盖矩阵与 Dream3R v0.3 落点

把 §2.1-§2.6 的 6 大子方向按综述 §10 抽象出的四轴 (轴 A 六类失败模式 / 轴 B 长序列内存四类 / 轴 C 测试时三类 / 轴 D 输出资产三类) 重新汇总, 得到 21 子类的覆盖矩阵 (per cycle 035 SOTA_MATRIX_V2 §6 + SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §3 first-class support 21 子类)。Dream3R v0.3 在矩阵上的落点是: ✓ first-class 6 子类 / ⚠ partial 11 子类 / ✗ absent 4 子类。

**✓ first-class 6 子类**: 轴 A 弱纹理 + 长基线 (C4 Critic Sampson + 共视 conflict); 轴 B B1 递推状态 (C2 StateToken) + B2 空间指针 (C2 AnchorBank) + B3 混合记忆 (C2 NSA + Mamba); 轴 D D1 4D pointmap (C1+C2+C3 输出)。

**⚠ partial 11 子类**: 轴 A 镜面 + 快速运动 + 尺度漂移 (C4 Critic 信号通道存在但阈值未按失败模式标定 — cycle 035 CRITIC_CALIBRATION_PLAN_V1 plan-only 待执行); 轴 B B4 预算治理 (帧预算接口存在但动态剪枝未对比 — cycle 035 LONG_SEQ_REAL_TABLE_PLAN B4 coverage gap); 轴 C C1 一致性优化 + C3 先验注入 (C4 Critic hybrid 含一致性元素但未拆为独立路径); 轴 D D2 dynamic mask (W18 输出 tensor 契约存在但实际 Permanence link 训练未跑); 等等 (具体逐项见 SOTA_MATRIX_V2 §6 Tables A-E)。

**✗ absent 4 子类**: 轴 A OOD (R-OOD-1 风险, 见 §9; CRITIC_CALIBRATION_PLAN_V1 A6 mode 计划但未启动); 轴 C C2 TTT 参数更新 (R 不存在专用风险条目, 但属于 §3 Q1 候选 v0.4 spec delta B1 范围); 轴 D D3 4DGS asset 渲染 (R-4DGS-LIC-1 + W27 gated); 输入扩展 axis (R-INPUT-EXT-1; 综述驱动优化提案 §3 bonus axis; v0.4 spec delta B3 候选)。

Dream3R v0.3 在四轴上的整体定位: **不押注单一支线**, 而是在 C1 Perceiver (DINOv3-S frozen backbone, 综述 §7 段 3 视觉特征 backbone 类) + C2 Memory (NSA three-branch 同时实装 B1+B2+B3) + C3 Permanence (Slot Attention 对应轴 D D2) + C4 Critic (Sampson + depth + 共视 三类信号 + repair actions; 含 C1 一致性优化元素但未拆) + C5 Composer (7 expert pool 含轴 A / 轴 B / 轴 D 各支线代表) + C6 Bus (CR-1..CR-6 cross-spec signal contract v2.1) 六模块上同时维持多机制并置。这一 no-all-in 设计 (per DEC-20260504-002) 是 §3 Q3 (多专家组合是否优于单一 expert) 的架构前提, 也是 §6 预期成果中"在 Q1 / Q2 / Q3 三组维度上提供候选架构层方案的实证依据"的可行性基础。

落点判断的工件锚点: SOTA_MATRIX_V2.md 五张 Tables A-E (cycle 035 P0-2 deliverable) + SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL.md §3 21 子类覆盖矩阵 (cycle 035 上游 proposal) + WORK_RISK_REGISTER.md v1.2 17 行 (cycle 035 +4 cross-spec + cycle 036 +3 proposal-cycle)。这些工件本章不复制, 仅引用; §5 消融与评测 / §9 研究风险将进一步展开各子类的 falsification 路径与风险缓解机制。

---

## §3 候选研究问题

本章在 §1.4 提出的三个核心研究问题 (Q1 验证机制 / Q2 长序列内存 / Q3 多专家组合) 基础上, 把每个 Q 与 §2 综述谱系 + §4 Dream3R v0.3 架构两侧的素材对齐, 给出 (a) 该研究问题的 gap identification, (b) Dream3R v0.3 在当前迭代中的候选路径, (c) falsification 路径在 §5 评测设计中的入口, (d) candidate-not-final 边界声明。三个 Q 之后, §3.4 阐明四个 finalist 模块 (Critic / Memory / Permanence / Composer) 的独立性 (per DEC-20260504-002 no-all-in), §3.5 给出本研究的整体研究地位声明 (per DEC-20260501-011 candidate-not-final)。

### 3.1 Q1 验证机制的架构层落地 (Critic 路径)

**研究问题陈述**: 在前馈式 3R 架构中, 几何验证 (无参数更新的一致性检查 + 修复) 与测试时适应 (TTT 风格的参数更新) 在架构层应否拆为两条独立路径? 拆分后两条路径的边际贡献能否分离?

**Gap identification**: 综述 §7 (映射到 §2.5) 区分了测试时三类机制 — C1 一致性优化 (Test3R, 无参数更新)、C2 测试时参数更新 (TTT3R)、C3 先验注入 (G-CUT3R / Pow3R / MASt3R-SfM)。现有 3R 系统中, 这三类机制以独立工作发表, 缺乏在单一架构内的并置评估。具体而言: Test3R 走 image-triplet 一致性优化, 不更新模型参数; TTT3R 把递推 3R 模型的状态更新当成在线 test-time training, 通过反向传播在推理阶段动态调整记忆更新速率。两者在性质上不同 (一个是固定模型 + 输出修正, 另一个是动态模型 + 参数更新), 但在哪一个 + 在何种条件下 + 边际贡献多大, 现有文献无法回答。

**Dream3R v0.3 候选路径**: 当前 C4 Critic (per SPEC-20260506-004 v0.2 §C4) 是"验证 + 修复"hybrid — 含 Test3R 风格的几何一致性验证 (Sampson / depth / 共视 conflict 三类信号, 无参数更新) + repair action 0/1/2 (rerun_local_region / rerun_global / A5 reroute_model 切换专家, 无参数更新) + repair action 3/4/5 stub (未实装)。TTT3R 风格的测试时参数更新路径未拆出。这一 hybrid 配置使得"几何验证 vs 参数适应"两类机制的边际贡献在 Dream3R v0.3 当前实装下无法分离。

为分离这两类机制, cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 B1 提出 v0.4 spec delta 候选: 在架构层把 C1 一致性优化与 C2 TTT 参数更新拆为两条独立路径, 让 C4 Critic 专注于 verification + repair, 另起一个 TTT 路径模块 (暂记 C4', 命名候选) 负责测试时参数更新。该 v0.4 spec delta 当前仍 proposal-status, 未在 cycle 039 起草; §3.1 在开题报告语义层把它作为本研究的明确实证缺口与候选设计方向。

**Falsification 路径**: 在 §5 实验设计中, ABL-v02-1..9 (SPEC-005 v0.2) 主要消融 v0.3 架构层 deltas, 不直接评估 C4 Critic 拆分前后的边际贡献。cycle 035 CRITIC_CALIBRATION_PLAN_V1 把六类失败模式 → C4 五个 sub-signal 的标定方案 ready, 但执行 gated on F-002 server authorization。Q1 的正面 evaluation 需要 (a) v0.4 spec delta B1 起草并实装 (post-开题报告 work), 然后 (b) 在 KITTI 长序列上对照"hybrid C4 v0.3" vs "拆分后 v0.4 (C4 + C4')"两组配置, 度量 route_regret + 修复后 pointmap L2 + 测试时算力开销。在开题报告时间窗口内 (M1-M5), Q1 的 evaluation 边界停留在 plan + spec-delta-drafted 阶段, 不承诺实证拆分对比。

**candidate-not-final 边界**: Q1 不主张 Critic 拆分路径优于 hybrid 路径; 也不主张 hybrid 路径优于 TTT3R 风格独立适应路径。Q1 是 "评估两条路径的边际差异是否在标准 3R 评测下显著存在" 的研究问题, 不是 "证明哪条路径更优" 的结论性命题。

### 3.2 Q2 长序列内存四类机制的架构层统一 (Memory 路径)

**研究问题陈述**: 在单一 3R 架构的 C2 Memory 模块内, 同时实装综述 §6 抽象出的长序列内存四类机制 (B1 递推状态 / B2 空间指针 / B3 混合记忆 / B4 缓存治理) 是否可行? 四类机制间的协同 / 冲突如何在长序列真实评测下显现?

**Gap identification**: 综述 §6 (映射到 §2.4) 把长序列 3R 工作分为四类互不互斥的内存机制支线。现有系统每个只占四类机制中的一档: CUT3R / STream3R / LongStream 走 B1 递推状态; Spann3R / Point3R 走 B2 空间指针; LONG3R / LoGeR / Mem3R 走 B3 混合记忆; OVGGT / PAS3R / FILT3R 走 B4 缓存治理。换言之, 四类机制在文献里是 disjoint 的; 一个系统选定一档之后, 其他三档在架构层缺席。这导致两个未回答的问题: (a) 单一架构能否同时实装四档而仍维持单帧 30-50 ms 帧预算 (per §4.1 Delta 1)? (b) 四类机制 jointly 实装时, 是否出现协同 (e.g., 空间指针 + 递推状态 的双通道写入提升长序列稳定性) 或冲突 (e.g., 缓存治理的剪枝策略 与混合记忆的注意力 weight 出现 contention)?

**Dream3R v0.3 候选路径**: C2 Memory (per SPEC-20260508-001 v0.3 addendum, supersedes Delta 3) 通过 NSA three-branch (compressed / selected / sliding) + AnchorBank K=256 + StateToken + Mamba-Transformer 混合循环结构覆盖前三档:

- B1 递推状态 ← StateToken (compressed branch, per NSA_MEMORY_INTEGRATION_MEMO §Compressed)
- B2 空间指针 ← AnchorBank K=256 (selected branch + selection gate)
- B3 混合记忆 ← Mamba hybrid + NSA three-branch (sliding branch 局部 + 全局)

B4 缓存治理 ⚠ partial: 帧预算约束接口存在 (per §4.1 Delta 1 latency budget 30-50 ms), 但动态剪枝策略未与基线对照 (per cycle 035 LONG_SEQ_REAL_TABLE_PLAN §B4 explicit coverage gap)。Q2 的核心是验证这一同时覆盖关系在 KITTI windows ≥ 10 长序列上是否 (a) 维持帧预算, (b) 在四类机制 jointly 实装时显现协同或冲突。

**Falsification 路径**: §5 实验设计将通过 ablate_recurrence.py 4 variants (baseline_cross_attention / mamba_hybrid / no_nsa / no_stable_memory) + cycle 035 LONG_SEQ_REAL_TABLE_PLAN 4 度量 (scale_drift_proxy / memory_decay_proxy / anchor_fill_rate / retrieval_diversity) 在 windows ∈ {10, 20, 50, 100} 上展开。Q2 的正面 evaluation 需要 F-002 server authorization 启动 KITTI 长序列 evaluation 跑; 在开题报告时间窗口内, Q2 的 evaluation 边界停留在 plan-ready + W17 实装完成阶段, 不承诺 KITTI 长序列实证数值。

B4 缓存治理子问题需要 v0.4 spec delta 候选 (动态剪枝接口) 或 v0.4 evaluation extension (现有帧预算接口的剪枝策略评估), 在开题报告时间窗口内不要求 closure。

**candidate-not-final 边界**: Q2 不主张 Dream3R v0.3 C2 Memory 是长序列内存机制统一的最终方案; 也不主张四类机制同时实装必然优于专一档。Q2 是 "评估单一架构同时实装多机制的可行性 + 协同/冲突显现" 的研究问题。

### 3.3 Q3 多专家组合是否优于单一 expert (Composer 路径)

**研究问题陈述**: 在前馈式 3R 架构中, 多专家组合 (best-of-N routing) 相对单一 expert 是否在标准评测下显现显著实证优势? 与 Test3R 内置 verifier 组合时, C4 Critic 的额外边际价值如何?

**Gap identification**: 综述 §3 + §6 + §7 + §8 (映射到 §2.2 / §2.4 / §2.5 / §2.6) 显示, 现有 3R 系统在不同 regime 上各有优势: MASt3R 在静态对几何上精度高, Fast3R 在多视图密集场景下高效, Spann3R 在流式场景下具备内置内存, CUT3R 在动态容忍场景下表现稳健, MoGe-2 在单目 pointmap 上 fail-safe, DepthAnything-V2 在单目 depth foundation 上 license-clear, Test3R 在测试时一致性验证上自带 verifier。多专家组合作为架构层方案 (per SPEC-20260506-004 v0.2 Delta 5 + COMPOSER_CAPABILITY_DESCRIPTORS 7-expert pool + DEC-20260507-001 Tier 1 in-pool) 在工程上具备直接合理性, 但其相对单一专家的实证优势是否在标准评测下显著存在, 现有文献缺乏对照实验。

**Dream3R v0.3 候选路径**: C5 Composer (per §4.6) 通过 7-expert pool + capability descriptor + 路由策略 (capability_match spread > 0 → cost_adjusted_match 解析 ties; spread = 0 → fail_fast 触发) 显式实装 best-of-N 路由。CR-1 与 C4 Critic 协作 (Critic A5 reroute 须有 Composer capability_match spread 支持); CR-4 处理 tied capability (Composer 不强制选择, Critic 在 epsilon_tie 窗口内决断)。

Q3 涉及两个子问题:

- **Q3-a**: best-of-N (7-expert pool) vs single-expert (e.g., MASt3R-only) 在 KITTI 真实数据上, pointmap L2 + route_regret + scale_drift_proxy 哪一组显著占优?
- **Q3-b**: 加入 Test3R 后, 7-expert pool + Test3R verifier 组合是否相对 6-expert pool + 外部 C4 Critic 显现额外边际价值? (per SPEC-20260507-002 v0.3 ABL-v02-10 Test3R-alone candidate)

**Falsification 路径**: §5 实验设计将通过 ABL-v02-4 (Composer best-of-N vs single-expert) + ABL-v02-6 (capability_match 测量) + ABL-v02-10 (Test3R-alone vs Test3R-in-pool) 三组消融在 KITTI / DTU 评测协议下展开。Q3 的正面 evaluation 需要 F-002 server authorization 启动多专家真实加载 + 多 regime workload 评测; 在开题报告时间窗口内 (W19-W23 期间), Q3 的 evaluation 边界停留在 ABL 实验组 ready + 评测协议 ready 阶段, 不承诺多专家路由 best practice 的最终判定。

**candidate-not-final 边界**: Q3 不主张 Dream3R v0.3 C5 Composer 7-expert pool 是多专家组合的最终方案; 也不主张 best-of-N 路由必然优于单一 expert。Q3 是 "提供 best-of-N vs single-expert 的对照实验证据" 的研究问题, 不是 "证明 best-of-N 普遍占优" 的结论性命题。

### 3.4 四个 finalist 模块的独立性 (no-all-in 设计)

per DEC-20260504-002 (no-all-in on single finalist), Dream3R v0.3 的 4 个 finalist 模块 (C4 Critic / C2 Memory / C3 Permanence / C5 Composer) 设计上保持独立性: 任一模块的实证表现不达标均不影响其他三模块的独立评估。这一独立性体现在三个层面:

- **架构层独立**: C4 / C2 / C3 / C5 各自有 standalone SPEC (SPEC-20260503-001 Critic / SPEC-20260503-002 Memory / SPEC-20260503-003 Permanence / SPEC-20260504-001 Composer); 跨模块通过 C6 Bus + CR-1..CR-6 cross-spec signal contract (per §4.7) 协同, 不通过 shared parameter 耦合。
- **评测层独立**: §5 实验设计中, ABL-v02-1..9 + ABL-memory-0..11 + CRITIC_CALIBRATION_PLAN_V1 + LONG_SEQ_REAL_TABLE_PLAN 各自针对单一模块设计消融组与度量, 不要求多模块 jointly 达标。
- **研究问题独立**: §3.1 Q1 (Critic 验证) / §3.2 Q2 (Memory 长序列) / §3.3 Q3 (Composer 路由) 各自挂在不同 finalist 模块上; Q1 失败不否决 Q2 / Q3, 反之亦然。

值得注意: §3.1-§3.3 的三个 Q 直接挂在 C4 / C2 / C5 三个 finalist 模块上, C3 Permanence (动静分离) 不构成本研究三个 Q 的核心命题, 但作为 4 finalist 之一仍在 §4.4 + §5 + §7 实证轨道内独立评估 (per DEC-20260504-002 no-all-in 不允许 silent retiring of any non-finalist track)。

这一独立性是 candidate-not-final 框架的工程支撑: 任一模块若被后续工作替换或修订, 其他模块仍可独立留存; 任一 Q 若被实证证伪, 其他两 Q 仍可独立评估。

### 3.5 候选 vs 最终的研究地位声明

per DEC-20260501-011 (Dream3R thesis reframe; candidate-not-final), Dream3R 是被评估的候选架构, 非项目收敛方案。这一研究地位声明对本研究三个 Q 与三个创新点 (§6.2) 的措辞构成硬约束:

- 本研究 **不** 论证 Dream3R 相对 SOTA 在任一单一指标上具有压倒性优势 (per §4.8 整体定位)
- 本研究 **不** 主张 Dream3R 是最终方案; v0.4 spec delta 候选 (B1 Critic 路径拆分 / B2 输出资产契约 / B3 输入扩展 axis, per cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5) 表明后续版本明确可能修订或替换 v0.3 架构
- 本研究 **不** 把 C2 Memory v0.3 (NSA + AnchorBank + StateToken + Mamba hybrid) 等同于长序列内存机制的最优方案; SPEC-20260508-001 v0.3 addendum 自身的 "未实装的 candidate" 与 "engineering-judgment 标记" 是 candidate 框架的实证体现

研究地位的正面表述: 本研究的成果是 (a) 一个具体的 3R 候选架构 (Dream3R v0.3) 的设计文档 + 实证评测, (b) 综述四轴判断与候选架构 X 路径的覆盖矩阵 (per §2.7 + cycle 035 SOTA_MATRIX_V2), (c) 多机制并置评估的 best practice 实证数据, (d) v0.4 spec delta 演进路径的明确候选清单。这些成果都不是 "Dream3R 是 3R 方向的最终解" 的结论性主张, 而是 "Dream3R v0.3 提供了一个具体且可证伪的候选架构 + 实证轨道" 的研究地位声明。

候选架构演化路径在 §8 时间安排 + §9 风险章节进一步展开, 包括 v0.4 spec delta 候选清单 + 候选架构被修订/替换的风险条目 (R-PROP-CLAIM-1 in WORK_RISK_REGISTER v1.2)。

---

## §4 Dream3R v0.3 架构

本章在 SPEC-20260506-004 v0.2 (架构 v0.2 delta, 六 Delta 设计) + SPEC-20260508-001 v0.3 (C2 Memory v0.3 addendum) + paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md v2.1 (CR-1..CR-6 cross-spec 信号契约) 的基础上, 按 8 个子节描述 Dream3R v0.3 的整体设计与 C1-C6 六模块。本章为开题报告层的架构描述, 不复述 SPEC 工程细节; SPEC 由 section / Delta / line anchor 引用, 不修改。

### 4.1 整体设计与帧预算

Dream3R v0.3 是按 architecture-first 主线 (per DEC-20260506-001) 设计的前馈式 3R 候选架构。整体框架由六个模块 (C1 Perceiver / C2 Memory / C3 Permanence / C4 Critic / C5 Composer / C6 Bus) 通过 cross-spec 信号契约 (CR-1..CR-6, v2.1) 串接, 接受图像序列 (或单图 / 图像对) 作为输入, 输出 4D pointmap (D1) + dynamic mask (D2) 张量契约 (W17-W18 cycle 034 完成); 可渲染 4DGS asset (D3) 由 W27 candidate gated。

per SPEC-20260506-004 v0.2 Delta 1, 整体设计采用 speed priority 与 frame budget 约束: 单帧目标延迟 30-50 ms (paper-derived from DINOv3 family throughput; engineering-judgment for 实际剩余 budget 分配到 C2-C6)。该约束是 §4.2 C1 backbone tier 选择 (Delta 2: DINOv3-S 取代 ViT-L) 与 §4.6 C5 Composer pool 7-expert 准入标准 (Delta 5: 排除 VGGT ~1.2B + MapAnything 因超 budget) 的源头驱动条件。

Dream3R v0.3 的 architecture novelty 集中在两条 pillar (per SPEC-004 v0.2 Delta 6 main claim narrowing):

- **Pillar A: Verification-as-architecture** — C4 Critic 把几何冲突检测与修复显式作为架构组件 (而非测试时附加路径), 通过 Sampson / depth / 共视三类信号 + repair actions 0-5 在每个窗口提供 falsifiability 与可观察的修复行为。
- **Pillar D: Heterogeneous best-of-N Composer** — C5 Composer 通过 7-expert pool + capability descriptor 显式路由, 每个 expert 在不同 regime (静态对 / 多视图 / 流式 / 单目 / 测试时一致性) 有差异化 capability_match。

其他模块 (C1 / C2 / C3 / C6) 是 enabling layer, 为 pillar A + D 提供输入特征、长序列上下文、动静分离、与跨模块信号路径; 它们各自的设计不构成 main claim。

### 4.2 C1 Perceiver (DINOv3-S frozen backbone)

C1 Perceiver 负责从输入图像 (或图像序列) 提取视觉特征, 供下游 C2-C5 消费。per SPEC-004 v0.2 Delta 2, C1 backbone 从 v0.1 的 ViT-L 替换为 DINOv3-S (S = Small tier, 自监督预训练 + frozen)。这一替换的依据是 DINOV3_C1_INTEGRATION_MEMO 中记录的两项 paper-derived metric: 参数量减少 ~14× (ViT-L ~300M → DINOv3-S ~21M), 推理延迟加速 ~5× (在标准 image-pair 输入下)。

frozen-backbone 决策 (而非 fine-tune) 的依据: (a) DINOv3 自监督训练已在大规模图像上学到通用 cross-view 对应能力 (类似综述 §3 提到的 CroCo → DUSt3R 迁移路径); (b) frozen 路径降低训练 cost 并避免 backbone-drift 风险; (c) 头部 (heads) 从头训, 在小型架构上更灵活。-B (Base) tier 作为 fallback 记录在 memo, 但 v0.3 默认不切换。

C1 输出特征 token 序列, 直接进入 C2 Memory 的三分支稀疏注意力 + C5 Composer 的路由层 + C4 Critic 的几何信号通道; 这些下游消费由 C6 Bus 的 cross-spec 信号契约 (per §4.7) 规约。

### 4.3 C2 Memory (NSA three-branch + AnchorBank + StateToken + Mamba hybrid)

C2 Memory 是 Dream3R v0.3 在长序列轴上覆盖综述 §6 四类机制 (B1 递推状态 / B2 空间指针 / B3 混合记忆 / B4 缓存治理) 的核心模块。per SPEC-20260508-001 v0.3 addendum (supersedes v0.2 Delta 3), C2 现行设计是 NSA-style three-branch sparse attention 与显式 AnchorBank + StateToken 的组合:

- **Compressed branch**: 把历史窗口的 token 流压缩为 fixed-size latent (对应综述 B1 递推状态; 实装由 StateToken 承载, 在每个窗口 incremental 更新; per NSA_MEMORY_INTEGRATION_MEMO §Compressed)。
- **Selected branch**: 通过 attention selection 从 AnchorBank (K=256 capacity) 中按 query relevance 抽取 top-k 三维空间锚点 (对应综述 B2 空间 / 指针记忆; 类似 Spann3R 的可寻址空间存储; per NSA_MEMORY_INTEGRATION_MEMO §Selected)。
- **Sliding branch**: 局部窗口的 frame-value tokens 直接保留 (对应综述 B3 混合记忆的局部窗口分量); 与 compressed + selected 通过 attention 共同消费 (per NSA_MEMORY_INTEGRATION_MEMO §Sliding)。

W17 Mamba-Transformer 混合循环结构 (cycle 034) 把上述三分支封装为可选择 recurrence backbone, 与 baseline cross-attention 在 ablate_recurrence 的 4 variants (baseline_cross_attention / mamba_hybrid / no_nsa / no_stable_memory) 中并置。B4 缓存治理 (动态剪枝 / 帧预算约束下的 anchor 替换) 是 partial coverage: 帧预算约束接口存在但动态剪枝策略未与基线对照 (per cycle 035 LONG_SEQ_REAL_TABLE_PLAN §B4 explicit coverage gap; 该 gap 是 §5 消融评测的 falsification 目标之一)。

C2 模块向 C6 Bus 发布的关键信号包括 `latent_drift_proxy` (per CR-3, 信息性, 不直接 gate C4 Critic 的 A5 reroute), 以及 `write_value_estimate` (per CR-2, 被 C3 Permanence 的 suppress_static_write 约束)。

### 4.4 C3 Permanence (Slot Attention + permanence_link)

C3 Permanence 负责动静分离与长序列上的 object identity 维护, 对应综述 §5 + §8 的 D2 dynamic mask 资产。模块以 Slot Attention 为骨架, 每个 slot 绑定一个 (类) 三维对象的 latent 表示; 通过 permanence_link (跨窗口的 slot 关联) 维持 identity 持续性。

C3 向 C2 Memory 发布 `suppress_static_write(r)` 信号: 当 Permanence 判定区域 r 为动态时, Memory 的 A2 (static map update) 必须遵守该 suppression (per CR-2)。这条 binding 是 Permanence-Memory 边界的关键 contract; Memory 若因结构限制无法遵守, 须显式 log `cross_spec_refusal` 并 surface 到 Advisor, 不允许 silent override。

C3 的设计与 MonST3R / Easi3R / RayMap3R 等动态主干路径 (per §2.3) 同属"动静分离"类机制, 但实现路径不同: MonST3R 类是动态主干 + 动态置信度回归, Dream3R v0.3 是静态优先 + Permanence-辅助。这两类方向在系统层互补而非互替, 是 §3 Q3 多专家组合的设计前提之一。

### 4.5 C4 Critic (Sampson + depth + 共视 conflict + repair actions)

C4 Critic 是 Pillar A (Verification-as-architecture) 的载体, 也是 §3 Q1 (验证机制路径) 的核心架构组件。per SPEC-20260506-004 v0.2 §C4, C4 通过三类几何信号检测当前窗口的几何冲突:

- **Sampson 几何**: 跨视图几何 epipolar 残差 (检测 pose / 内参与匹配 不一致)。
- **Depth 一致性**: 跨视图 depth 重投影残差 (检测 scale / depth 漂移)。
- **共视 conflict**: 多视图共视区域的 pointmap 一致性 (检测局部 mismatch)。

三类信号聚合为 `conflict_score(t)`, 通过 threshold `theta_conflict` 触发 repair actions:

- Action 0: no_action (conflict_score 低于 threshold)
- Action 1: rerun_local_region (局部 region 重跑 C1 + C2)
- Action 2: rerun_global (全窗口重跑)
- Action 3-5: stub (predicted / 未实装; per spec v0.3 行动空间扩展候选)

A5 reroute_model 是与 C5 Composer 协作的特殊 action: 当 conflict_score 高且 capability_match spread > 0, Critic 触发 reroute 切换到 Composer pool 的另一 expert (per CR-1 binding: Critic A5 requires Composer agreement on capability_match spread)。

C4 当前是"验证 + 修复"hybrid: 含 Test3R 风格的几何一致性验证 (无参数更新) + repair action 的输出修正 (无参数更新), 但不含 TTT3R 风格的测试时参数更新 (per cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 B1: 在架构层把 C1 一致性优化与 C2 TTT 参数更新拆为两条独立路径, 是 v0.4 spec delta 候选, 当前仍 proposal-status)。

per cycle 035 CRITIC_CALIBRATION_PLAN_V1, 三类信号的 threshold 阈值目前是统一 default, 未按六类失败模式 (弱纹理 / 镜面 / 快速运动 / 长基线 / 尺度漂移 / OOD) 逐类标定; 标定 plan 已 ready, 执行 gated on F-002 server authorization。

### 4.6 C5 Composer (7 expert pool + capability descriptor)

C5 Composer 是 Pillar D (Heterogeneous best-of-N) 的载体, 也是 §3 Q3 (多专家组合) 的核心架构组件。per SPEC-20260506-004 v0.2 Delta 5 + COMPOSER_CAPABILITY_DESCRIPTORS, 当前 expert pool 含 7 个 admitted lightweight expert:

| Expert ID | Model | innovation point | 规模 | attention regime |
|---|---|---|---|---|
| EXPERT-01 | MASt3R | pair / matching head | ~300M | full attn |
| EXPERT-02 | Fast3R | many-view single fwd | ~580M | full attn |
| EXPERT-03 | Spann3R | streaming spatial anchor | ~250M | (spatial) |
| EXPERT-04 | CUT3R | online persistent state | ~300M | (recurrent) |
| EXPERT-05 | MoGe-2 | mono pointmap | ~200M | full attn |
| EXPERT-06 | DepthAnything-V2 | mono depth foundation | ~25M Small | (depth prior) |
| EXPERT-07 | Test3R | test-time consistency verification | (lazy off-path; backbone + iteration) | full attn |

排除依据 (per Delta 5 engineering-judgment): VGGT (~1.2B; 超 frame budget) / MapAnything (multi-modal foundation, 太重) / PE Perception Encoder (太重) / Kimi Linear KDA (LM-to-3R transfer 不追求, RU-007 历史保留)。

Composer 的路由策略基于 capability descriptor: 每个 expert 在 (输入 regime, 输出 schema, infrastructure cost, attention regime, capability_match, failure modes) 9 axes 上携带 paper-derived / engineering-derived 标签; 路由层按当前输入 regime 与 capability_match 计算 spread, 当 spread > 0 时由 cost_adjusted_match (per v2 contract upgrade, DEC-20260504-004) 解析 ties; 当 spread = 0 时 fail_fast 触发 (per SPEC-20260504-001 §fail_fast_threshold)。

C5 与 C4 Critic 通过 CR-1 协作 (Critic A5 reroute 须有 Composer capability_match spread 支持); 通过 CR-4 处理 tied capability (Composer 不强制选择 top-1, 由 Critic 在 epsilon_tie window 内按 Critic-internal preference 决断)。

### 4.7 C6 Bus (CR-1..CR-6 cross-spec signal contract v2.1)

C6 Bus 是 Dream3R v0.3 跨模块协同的信号契约层, 不是 trainable 模块。per paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md v2.1 §Conflict Resolution Rules, 六条规则规约模块间的争议解决与信号 binding:

- **CR-1**: Critic A5 reroute_model 要求 Composer capability_match spread > 0; Critic 不发明 Composer 未刻画的模型。
- **CR-2**: Permanence suppress_static_write(r) 对 Memory 的 A2 是 binding; 不能 silent override; 若结构限制无法 honor, 须 log cross_spec_refusal + surface Advisor。
- **CR-3**: Memory latent_drift_proxy 是 Critic 验证的信息性输入, 不直接 gate A5; drift 单独不构成 Critic reroute 条件 (Critic 与 Memory falsification 独立)。
- **CR-4**: Composer top-1 / top-2 capability_match 在 epsilon_tie (default 0.05; inferred) 窗口内时, Composer 不强制选择, 由 Critic 按 Critic-internal preference 决断 (v2 起以 cost_adjusted_match 为 canonical, 处理 cost 不对称 tie)。
- **CR-5**: 所有 cross-spec 信号携带 producer 的 evidence label, 沿信号路径传播; Critic 不能 silently 把 inferred 信号升级为 paper-proven (per RESEARCH_CODE_DISCIPLINE.md rule 5 Honesty Override)。
- **CR-6**: 每个 cycle 009 case card 必须记录消费的 cross-spec 信号 + producer evidence label at consumption time; unknown 信号须 caveat。

v2.1 的 additive 变化 (per DEC-20260505-001) 是新增 "Forward-reference null protocol" 子节, 把 cycle 009 + cycle 010 case card 中已使用的"signal 在被消费时还未发布, 消费方回 null + 留 forward reference"模式正式化, 不修改 v2 substance。CR-7 (external_prior_conflict) 作为 v2.2 候选已记录在 cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 C2, 仍 proposal-status。

### 4.8 与现有 3R 系统的结构差异

把 Dream3R v0.3 的六模块设计与 §2 综述谱系对比, 主要结构差异在三点:

**结构差异 1 (轴 B 长序列内存): 单一架构同时覆盖多类机制**。综述 §6 中 CUT3R / Spann3R / LONG3R / Point3R 等系统各自占四类机制 (B1-B4) 的一档; Dream3R v0.3 的 C2 Memory (NSA three-branch + AnchorBank + StateToken + Mamba hybrid) 在单一架构内同时实装 B1 + B2 + B3, B4 partial。这一同时覆盖关系是 §3 Q2 (长序列内存机制统一) 的实证目标。

**结构差异 2 (轴 C 测试时机制): 显式区分"验证"与"路由切换", 暂未区分"参数更新"**。综述 §7 中 Test3R / TTT3R / G-CUT3R / Pow3R 等系统在测试时三类机制中分散存在; Dream3R v0.3 通过 C4 Critic 显式实装一致性验证 + repair action, 通过 C5 Composer 显式实装 expert reroute, 通过 CR-1 显式协调两者; 但当前 C4 是 verification-only (无参数更新), TTT 类参数更新路径未拆出 (B1 v0.4 spec delta 候选)。这一未拆分构成 §3 Q1 (验证机制路径) 的研究问题。

**结构差异 3 (轴 D + 多专家组合): 显式 best-of-N expert pool**。综述 §3-§7 中各 3R 系统在不同 regime 上分别擅长 (MASt3R 静态对 / Fast3R 多视图 / Spann3R 流式 / CUT3R 动态容忍 等); Dream3R v0.3 的 C5 Composer 通过 7-expert pool + capability descriptor 显式 best-of-N 路由, 把这一 regime 分化变成架构层显式选择。这一设计是 §3 Q3 (多专家组合实证评估) 的架构前提, 也是 cycle 029 review (per MEMORY_V03_ABLATION_REVIEW.md R-029-1..5) 之后保留的 cycle 035 SOTA_MATRIX_V2 §6 first-class support 6 子类的来源之一。

整体定位: Dream3R v0.3 不主张相对现有 SOTA 在任一单一指标上压倒性领先, 而是评估"在统一架构内同时实装多机制 + 显式路由"这一架构层 best practice 在 Q1 / Q2 / Q3 三组维度上的实证表现 (per DEC-20260501-011 candidate-not-final + DEC-20260504-002 no-all-in)。架构与单一支线方法的实证对照将在 §5 消融与评测设计中按 ABL-v02-1..9 + ABL-memory-0..11 + 综述驱动新增 calibration / long-seq 评测展开。

---

## §5 消融与评测设计

<!-- TBD cycle 040; 起草目标 ~2800 字 -->
<!-- 上游素材: SPEC-20260506-005 v0.2 ABL-v02-1..9 + SPEC-20260507-002 v0.3 ablation addendum + SPEC-20260508-002 ABL-memory-0..11 + CRITIC_CALIBRATION_PLAN_V1 + LONG_SEQ_REAL_TABLE_PLAN -->
<!-- 子节建议: 5.1 三层证据阶梯 + 5.2 ABL-v02-1..9 架构消融 + 5.3 ABL-memory-0..11 记忆消融 + 5.4 Critic 阈值校准 (CRITIC_CALIBRATION_PLAN_V1) + 5.5 长序列真实表 (LONG_SEQ_REAL_TABLE_PLAN) + 5.6 评测数据集 (KITTI / DTU 拟扩展) + 5.7 指标 (pointmap L2 + route_regret + scale_drift + memory_decay) -->

---

## §6 预期成果

本章按三个子节给出本研究的预期成果与创新点。§6.1 列出本研究的预期交付物 (架构设计文档 + 原型实现 + 评测结果 + 综述与方法学副产物); §6.2 把 §3 三个 Q (Q1 验证 / Q2 长序列内存 / Q3 多专家组合) 对应的三个创新点 (IP1 verification-as-architecture / IP3 NSA-hybrid memory / IP2 heterogeneous best-of-N Composer) 显式声明; §6.3 阐明本研究与现有工作的实证差异 — 本研究 **不** 主张 Dream3R 相对 SOTA 压倒性领先, 主张提供多机制并置的对照实验证据。三个子节都受 STYLE_CONTRACT §5 candidate-not-final 句式表硬约束。

### 6.1 预期交付物

本研究的预期交付物分四类:

**架构设计文档**: Dream3R v0.3 的完整 SPEC 系列 (已有 SPEC-20260506-004 v0.2 + SPEC-20260508-001 v0.3 + SPEC-20260507-001 v0.2 + SPEC-20260507-002 v0.3 + SPEC-20260506-005 v0.2 + SPEC-20260508-002 + paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md v2.1 七份), 加上 cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL §5 提出的 v0.4 spec delta 候选 (B1 Critic 路径拆分 / B2 输出资产契约 / B3 输入扩展 axis), 在开题报告时间窗口结束 (M5 节点) 之前形成 v0.3 完整稿 + v0.4 spec delta 候选清单。

**原型实现**: code/dream3r/ (服务器部署 /hdd3/kykt26/code/dream3r/), 当前 W1-W18 完成 (per RECENT_PROGRESS.md), 包括 DINOv2 backbone 实际跑通 + 3D-aware retrieval + active/stable state + Grassmannian 正则化 + 几何 Critic + ISA slot + 真实 MASt3R + Spann3R adapter + W17 Mamba-Transformer 混合循环 + W18 GaussianHead tensor 契约 (renderer-free)。开题报告时间窗口内的预期延伸: W19 多专家真实加载 + W20 路由层实装 + W21-W22 ablate_recurrence + critic_calibration 真实数据扩展。W23-W30 (DTU loader / TTT / 4DGS renderer / 真实数据训练) 作为 post-开题报告 work, 不在预期交付物声明范围内。

**评测结果**: ABL-v02-1..9 + ABL-memory-0..11 + cycle 035 CRITIC_CALIBRATION_PLAN_V1 + LONG_SEQ_REAL_TABLE_PLAN 在 KITTI 真实数据上的对照实验数据集合, 包括 (a) §3.1 Q1 verification 路径的 evaluation gating data (拆分前 hybrid v0.3 实证基线), (b) §3.2 Q2 长序列内存四类机制在 windows ∈ {10, 20, 50, 100} 上的协同/冲突显现数据, (c) §3.3 Q3 best-of-N vs single-expert 的 pointmap L2 + route_regret + scale_drift_proxy 对照数据, (d) Test3R-alone 在 pillar A 上与 Dream3R Critic-gate pipeline 的对照数据 (per ABL-v02-10)。所有评测结果都以"评估候选架构 X 是否在 ... 维度上呈现优势"的句式呈现, 不以"宣称 X 优于 SOTA"的句式呈现。

**综述与方法学副产物**: Track B 3R-mix 中文综述 (18 A4 页 / 44 引文 / 6 图 5 表 / 2026-05-15 prose naturalization deliverable, arXiv 自存档路线) 是本研究的方向定调副产物, 已在 cycle 036 packaging 完成 SHA256 pre-fill。Track A 与 Track B 共享 references.bib 但不互相引用 (sibling artifacts; 词汇隔离声明 in RELATION_TO_TRACK_A_2026-05-16.md)。cycle 035 SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL + SOTA_MATRIX_V2 + CRITIC_CALIBRATION_PLAN_V1 + LONG_SEQ_REAL_TABLE_PLAN 4 markdown deliverables 是综述反哺 Track A 主线的实证副产物。

### 6.2 创新点声明

per §3.1-§3.3 三个 Q 与 §4 六模块设计, 本研究的三个创新点 (Innovation Points, IP) 如下:

**IP1: Verification-as-architecture (校验作为架构组件; 对应 Q1 + §4.5 + Pillar A)**

把几何验证 (一致性检查 + 修复) 显式作为 3R 架构层组件, 而非测试时附加路径或后处理步骤。Dream3R v0.3 通过 C4 Critic (Sampson / depth / 共视 conflict 三类信号 + repair actions 0/1/2 stub 3/4/5) 实现这一架构组件化。creation novelty 不在于"几何验证本身是新概念"(经典 BA / SfM 也含验证元素), 而在于"把验证 + 修复在前馈式 3R 架构内显式作为模块 + 信号契约的一部分"。本研究的对照实验将提供 C4 Critic 在 Dream3R v0.3 内的 (a) 失败模式检出率, (b) 修复动作有效性, (c) 与 Test3R 内置 verifier 在测试时一致性优化上的对照数据。

句式约束: 不说 "Dream3R 已完全解决几何验证问题"; 说 "Dream3R 在架构层提供了几何验证的候选模块设计 + 对照实验证据"。

**IP2: Heterogeneous best-of-N Composer (异构多专家组合; 对应 Q3 + §4.6 + Pillar D)**

把多个不同 regime 优势的 3R expert (MASt3R / Fast3R / Spann3R / CUT3R / MoGe-2 / DepthAnything-V2 / Test3R) 通过 capability descriptor + 路由策略组合, 在架构层显式实装 best-of-N 选择。creation novelty 不在于 "best-of-N 路由本身" (经典 ensemble 也含 best-of-N 元素), 而在于 (a) capability descriptor 把每个 expert 在 9 axes 上携带 paper-derived / engineering-derived 标签, (b) 路由由 capability_match spread + cost_adjusted_match + epsilon_tie 决定 (per v2 contract upgrade, DEC-20260504-004), (c) 与 C4 Critic 通过 CR-1 协作路由切换。本研究将提供 best-of-N (7-expert pool) vs single-expert 在 KITTI 真实数据上的 pointmap L2 + route_regret 对照数据。

句式约束: 不说 "best-of-N 路由必然优于单一 expert"; 说 "本研究提供 best-of-N vs single-expert 的对照实验数据"。

**IP3: NSA-hybrid memory (统一覆盖长序列内存四类机制; 对应 Q2 + §4.3 + 综述 §6 四类)**

把综述 §6 抽象出的长序列内存四类机制 (B1 递推状态 / B2 空间指针 / B3 混合记忆 / B4 缓存治理) 在单一 C2 Memory 模块内通过 NSA three-branch + AnchorBank K=256 + StateToken + Mamba-Transformer 混合循环结构 jointly 实装。creation novelty 不在于"NSA 三分支机制本身" (NSA 类机制在文献中已有), 而在于 (a) 把 NSA 三分支映射到长序列内存四类机制的前三档 (compressed ↔ B1, selected ↔ B2, sliding ↔ B3), (b) 同时维持单帧 30-50 ms 帧预算 (per Delta 1), (c) B4 缓存治理 partial coverage 与实证缺口的显式承认 (per cycle 035 LONG_SEQ_REAL_TABLE_PLAN §B4 coverage gap)。本研究将提供 ablate_recurrence 4 variants 在 KITTI windows ∈ {10, 20, 50, 100} 上的 scale_drift_proxy + memory_decay_proxy + anchor_fill_rate + retrieval_diversity 数据。

句式约束: 不说 "Dream3R 已完全解决长序列内存问题"; 说 "Dream3R 在单一架构内同时实装四类机制中的前三档, 为长序列内存机制统一提供候选实证"。

三个 IP 与 §3.1-§3.3 三个 Q 一一对应 (Q1 ↔ IP1; Q2 ↔ IP3; Q3 ↔ IP2)。三个 IP 的 candidate-not-final 共同体现在 "提供 ... 候选 ... + 对照实验证据" 的句式模式, 而非 "宣称 X 优于 Y" 的结论性命题。

### 6.3 与现有工作的实证差异

本研究与现有 3R 工作的实证差异不在 "Dream3R 相对 SOTA 在某单一指标上压倒性领先" — 这一论断超出本研究的 candidate-not-final 边界, 也与 §4.8 整体定位 + DEC-20260501-011 + DEC-20260504-002 三项决策矛盾。

本研究与现有 3R 工作的实证差异在三个层面:

- **方法学差异**: 现有 3R 工作以 "单一论文一个方法" 的离散发表模式为主; 本研究以 "在统一架构内多机制并置评估" 的对照实验模式为主。前者擅长在某一档 (e.g., B1 递推状态 或 B2 空间指针) 上 push SOTA; 后者擅长在多档之间提供协同 / 冲突 / 边际贡献的对照数据。两者不互替, 是研究方向上的互补关系。
- **失败模式系统化差异**: 现有 3R 工作 (per 综述 §10) 多数把六类典型失败模式作为 "limitations" 章节简要提及; 本研究通过 C4 Critic + cycle 035 CRITIC_CALIBRATION_PLAN_V1 把六类失败模式映射到 5 个 sub-signal 的逐类阈值标定, 提供失败模式系统化的对照实验证据。这一差异是 IP1 的方法学体现。
- **架构组合差异**: 现有 3R 工作以 "single best architecture" 为主; 本研究通过 C5 Composer + COMPOSER_CAPABILITY_DESCRIPTORS 7-expert pool 显式实装 "heterogeneous best-of-N"。这一差异是 IP2 的架构体现。

本研究的实证目标不是把 Dream3R 推上 KITTI / DTU / ScanNet 等单一 leaderboard 的 top-N; 而是在 §5 实验设计指定的 evaluation 协议下, 提供三个 Q 与三个 IP 对应的对照实验数据, 让后续工作 (无论是 Dream3R v0.4 演进, 还是其他 3R 架构) 可以基于这些数据判断: (a) 当前 Dream3R v0.3 在三个 Q 上的覆盖与缺口, (b) v0.4 spec delta 候选 (B1 / B2 / B3) 应优先推进哪一个, (c) 是否有 v0.5 候选架构 需要替换 v0.3 整体设计。这一对照实验数据 + 候选演化路径是本研究的核心实证差异, 也是 candidate-not-final 框架的工程落点。

---

## §7 已完成工作

<!-- TBD cycle 040; 起草目标 ~2200 字 -->
<!-- 上游素材: code/dream3r/RECENT_PROGRESS.md W1-W18 + KITTI smoke + cycle 033-035 deliverables + 综述 deliverable -->
<!-- 子节建议: 7.1 架构设计 (SPEC 系列 v0.1/v0.2/v0.3) + 7.2 实现里程碑 W1-W18 + 7.3 KITTI 真实数据集成证据 + 7.4 综述发布 (Track B) + 7.5 综述反哺主线 (cycle 035 4 deliverables) + 7.6 cycle 历史 (cycle 015 / 016 / 018 / 019 / 020 / 021 / 022 / 023 / 024 / 025 / 026 / 027 / 028 / 029 / 030 / 031 / 032 / 033 / 034 / 035) -->

---

## §8 时间安排

<!-- TBD cycle 040; 起草目标 ~1500 字 -->
<!-- 上游素材: code/dream3r/NEXT_PHASE_ROADMAP.md W19-W27 + 综述驱动优化提案 §6 重排 + DEC-20260515-001 §Next Direction A-E -->
<!-- 子节建议: 8.1 短期 (cycle 036-041 开题报告起草) + 8.2 中期 M3-M5 (W19-W23 真实路由 + W24 Critic 校准 + W25 TTT + W26 输入扩展 + B1/B2/B3 v0.4 spec delta) + 8.3 长期 M6-M8 (W27 3DGS renderer + 真实数据训练 + 论文撰写 + 综合评测) -->

---

## §9 研究风险

<!-- TBD cycle 041; 起草目标 ~1500 字 -->
<!-- 上游素材: WORK_RISK_REGISTER v1.2 17 rows + cycle 036 +3 = 20 行总计 -->
<!-- 子节建议: 9.1 域外检测缺口 (R-OOD-1) + 9.2 外部 prior 冲突 (R-EXT-PRIOR-1) + 9.3 4DGS license 链 (R-4DGS-LIC-1) + 9.4 输入扩展 axis 空缺 (R-INPUT-EXT-1) + 9.5 算力约束 (F-002) + 9.6 双稿语义漂移 (R-PROP-SYNC-1; 元层风险) + 9.7 候选架构被替换的可能性 (candidate-not-final 内在风险) -->

---

## 双稿同步状态

| Sync 时间 | Cycle | 章节 | 同步方向 | Vocab 替换条目 | Vocab 防火墙 grep 验证 |
|---|---|---|---|---|---|
| 2026-05-16 | 036 close | §1 first draft | internal → external (initial) | 13 (per STYLE_CONTRACT §2 seed) | DRAFT_EXTERNAL §1 grep clean (0 hits on cycle/SPEC-/DEC-/CR-N/agent/skill/workflow/本地项目 + 0 hits on Dream3R case-insensitive) |

cycle 037+ 起草 §2 后追加。

---

## 元数据

| 字段 | 取值 |
|---|---|
| 文件路径 | `Dream/planning/proposal_dream3r/DRAFT_INTERNAL_V1.md` |
| 创建日期 | 2026-05-16 |
| 状态 | v1 draft, §1 complete, §2-§9 placeholder |
| 作者 | Dream agent (cycle 036) |
| 上游决策 | DEC-20260516-001 |
| 双稿关系 | master per STYLE_CONTRACT §3 规则 1 |
| 当前字数 | §1 约 1800 字 (符合 OUTLINE §2 内稿 §1 估算 ~2000 字, 偏差 < 10%) |
| 下游 | cycle 037 (§2 起草) → cycle 038 (§4) → cycle 039 (§3+§6) → cycle 040 (§5+§7+§8) → cycle 041 (§9+通稿审查) → cycle 042 (最终修订+提交) |

---

**End of DRAFT_INTERNAL_V1 §1.** 本文件是 cycle 036 P0-B 子任务 deliverable; cycle 037+ 按 OUTLINE_V1 §4 顺序逐章起草。
