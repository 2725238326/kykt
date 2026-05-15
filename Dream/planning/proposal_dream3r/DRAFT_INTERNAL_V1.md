# Dream3R 开题报告（内部稿 V1）

| 字段 | 取值 |
|---|---|
| 文件类型 | 开题报告内部稿 (Dream-vocabulary; 不外发) |
| 创建日期 | 2026-05-16 |
| 状态 | v1 draft; §1 完整起草; §2-§9 placeholder, 待 cycle 037+ 起草 |
| 授权根 | DEC-20260516-001 (cycle 036 launch) |
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

<!-- TBD cycle 037; 起草目标 ~4000 字 -->
<!-- 上游素材: 综述 §2-§9 全章 + SPEC-20260507-001 v0.2 Tier 1-5 + Axis 9-11 + SOTA_MATRIX_V2 + references.bib 44 entries -->
<!-- 子节建议: 2.1 基础谱系 (DUSt3R / MASt3R) + 2.2 多视角扩展 (Fast3R / VGGT / MapAnything) + 2.3 视频动态 4D (MonST3R / POMATO / Easi3R) + 2.4 长序列内存四类 (CUT3R / Spann3R / LONG3R / Point3R / Mem3R) + 2.5 测试时三类 (Test3R / TTT3R / G-CUT3R / Pow3R / MASt3R-SfM) + 2.6 输出资产三类 (4D pointmap / dynamic mask / 4DGS) + 2.7 综述四轴覆盖矩阵 + Dream3R 落点 -->

---

## §3 候选研究问题

<!-- TBD cycle 039; 起草目标 ~1800 字 -->
<!-- 上游素材: §1.4 三个核心研究问题展开 + DEC-20260501-011 + DEC-20260504-002 -->
<!-- 子节建议: 3.1 Q1 验证机制路径 (Critic) + 3.2 Q2 长序列内存路径 (Memory) + 3.3 Q3 多专家组合路径 (Composer) + 3.4 Critic / Memory / Permanence / Composer 4 finalist 模块的 no-all-in 设计 + 3.5 candidate-not-final 边界声明 -->

---

## §4 Dream3R v0.3 架构

<!-- TBD cycle 038; 起草目标 ~4000 字 -->
<!-- 上游素材: SPEC-20260506-004 v0.2 + SPEC-20260508-001 v0.3 + CROSS_SPEC_SIGNAL_CONTRACT v2.1 + COMPOSER_CAPABILITY_DESCRIPTORS + DINOV3_C1_INTEGRATION_MEMO + NSA_MEMORY_INTEGRATION_MEMO -->
<!-- 子节建议: 4.1 C1 Perceiver (DINOv3-S) + 4.2 C2 Memory (NSA + AnchorBank + StateToken) + 4.3 C3 Permanence (Slot Attention) + 4.4 C4 Critic (Sampson + depth + 共视) + 4.5 C5 Composer (7 expert pool) + 4.6 C6 Bus (CR-1..CR-6) + 4.7 与现有 3R 系统的结构差异 -->

---

## §5 消融与评测设计

<!-- TBD cycle 040; 起草目标 ~2800 字 -->
<!-- 上游素材: SPEC-20260506-005 v0.2 ABL-v02-1..9 + SPEC-20260507-002 v0.3 ablation addendum + SPEC-20260508-002 ABL-memory-0..11 + CRITIC_CALIBRATION_PLAN_V1 + LONG_SEQ_REAL_TABLE_PLAN -->
<!-- 子节建议: 5.1 三层证据阶梯 + 5.2 ABL-v02-1..9 架构消融 + 5.3 ABL-memory-0..11 记忆消融 + 5.4 Critic 阈值校准 (CRITIC_CALIBRATION_PLAN_V1) + 5.5 长序列真实表 (LONG_SEQ_REAL_TABLE_PLAN) + 5.6 评测数据集 (KITTI / DTU 拟扩展) + 5.7 指标 (pointmap L2 + route_regret + scale_drift + memory_decay) -->

---

## §6 预期成果

<!-- TBD cycle 039; 起草目标 ~1300 字 -->
<!-- 上游素材: DEC-20260501-011 + DEC-20260504-002 + §3 三个 Q -->
<!-- 子节建议: 6.1 架构设计文档 (SPEC v0.3/v0.4 系列) + 6.2 原型实现 (W1-W22 完成 + W23-W30 候选) + 6.3 评测结果 (Q1/Q2/Q3 实证) + 6.4 创新点声明 (verification-as-architecture / heterogeneous best-of-N / NSA-hybrid memory; 严格使用 candidate-not-final 句式) -->

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
