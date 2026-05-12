# 3R survey — handoff state

This file is the top-level handoff for the 3R survey project at `E:\kykt\Dream\3R-mix`.
It is updated after every substantive editing pass. For fine-grained history see `notes/work_log.md`.

## 当前目标

将 LaTeX 中文 3R / feed-forward 3D reconstruction 综述打磨成一篇结构清楚、证据可追溯、语言克制的综述论文。主稿为 `main.tex`，BibTeX 为 `references.bib`，PDF 输出 `build/main.pdf`。

## 当前稿件状态

- 框架：LaTeX（`ctexart`，`xelatex` 编译，`natbib`/`unsrtnat`）。已彻底放弃 Typst 路线，`main.typ` 仅作为历史快照保留，不再维护。
- 主稿：`main.tex`，最后更新 2026-05-13。
- PDF：`build/main.pdf`，13 页，可正常打开。
- 章节结构（共 10 节）：
  1. 引言
  2. 从传统几何流程到点图表示
  3. 基础谱系：DUSt3R、MASt3R 与 SfM 接口
  4. 多视角规模化与统一视觉几何
  5. 视频、动态场景与 4D 重建
  6. 长序列重建中的状态、记忆与缓存
  7. 测试时验证、修正与先验输入
  8. 从几何预测到可查看输出
  9. 应用证据、复现边界与失败样本记录
  10. 开放问题与结论
- 表格：5 张（`tab:foundation`, `tab:dynamic`, `tab:memory`, `tab:testtime`, `tab:application`），均使用 booktabs。
- 图：4 张（`fig:lineage` 谱系图、`fig:memory` 长序列记忆机制图、`fig:application` 应用路径图，外加 §2 末 TikZ 谱系图）。所有图均为内嵌 TikZ，无外部图片依赖。
- 参考文献：43 条，全部在正文中至少被引用一次，已移除 `\nocite{*}`。

## 本轮已完成修改（2026-05-13）

按章节、图、表、参考文献分类：

- **章节**：
  - 移除作者行 "KYKT Dream 调研组"，改为空。
  - 将旧 §3（DUSt3R/MASt3R + 多视角）拆为 §3（DUSt3R/MASt3R/SfM 接口）和 §4（多视角规模化与统一视觉几何）。
  - 新增 §9（应用证据、复现边界与失败样本记录），并把应用图和应用矩阵表移入。
  - 把旧 "讨论" + "结论" 两节合并为 §10（开放问题与结论）。
- **正文**：
  - 重写摘要，使其先讲问题分支组织线索，再列代表模型；显式提出三类证据（论文 / 官方仓库 / 本地复现）。
  - 引言"证据边界"段去掉"流程跑通 ≈ 质量领先"的措辞，改为"接口和依赖成立 ≠ 几何质量更优"。
  - 动态场景段去掉"通常被视为...重要代表"、"应更克制地理解"等评议化措辞，改为机制直述；末句明确"4D pointmap / dynamic mask / 4DGS asset"不是同一输出。
  - 长序列段从单段堆名改为四类（空间指针 / causal-autoregressive / hybrid memory / 预算治理与滤波），每类一句机制描述。
  - 测试时段拆为三段：Test3R vs TTT3R 区分、G-CUT3R vs Pow3R vs MASt3R-SfM 的先验进入位置、外部先验（Depth Pro / Metric3Dv2 / DINO / CoTracker / SpatialTracker / SAM2）的辅助角色与冲突风险。
  - 输出段（§8）梳理 Splatt3R / InstantSplat / NoPoSplat 的具体差异。
  - §9 显式定义"paper-proven / official-code / local-smoke-test / application-validated / 尚需确认"四级证据，并与 `fig:application` 和 `tab:application` 对齐。
  - §10 把原讨论的四点合并为一段开放问题，再以一段收束。
- **图**：仅 `fig:application` 因章节移位换位；TikZ 内容未改。
- **表**：
  - `tab:foundation` 内"领先"/"SOTA"措辞替换为"更优"/"按实验表引用"。
  - `tab:application` 同样把"领先"替换为"更优"，并把"系统报告"行的过度表述改为"质量已验证"。
  - **新增** `tab:testtime`（测试时机制与先验输入的进入位置和证据边界，9 行）。
- **参考文献**：删除 `\nocite{*}`；确认 43 个 bib key 都在正文显式引用，BibTeX 0 warning。

## 关键决策

- 继续使用 LaTeX，不回到 Typst。
- Easi3R 仅在动态 3R / training-free motion disentanglement 机制中讨论；不放在主线中心。
- 不直接嵌入论文 PDF 截图。所有图均为 TikZ 概念图。
- 应用证据矩阵明确标注"系统报告"行，把"流程跑通≠质量已验证"作为系统层规则写入正文。
- 正文中不出现 KYKT / Dream / Dream3R / agent / skill / workflow / 本地项目 等内部语境词（已 `Grep` 校验）。

## 证据边界

正文按四级证据标注：

- **paper-proven**：DUSt3R / MASt3R / Fast3R / VGGT / CUT3R / Spann3R / MonST3R / Test3R 等核心模型的"机制描述"段都来自论文 abstract/方法图，可放心引用。
- **official-code**：仅说明仓库存在，不构成质量结论；具体许可证、显存、依赖未全部独立核对。
- **local-smoke-test**：在正文中以"接口和依赖成立"形式出现，未与"质量更优"绑定。
- **尚需确认**：2026 年预印本（LoGeR、Mem3R、OVGGT、PAS3R、FILT3R、RayMap3R、LongStream）的代码状态、许可、实验表细节尚未独立核对；MV-DUSt3R+ 的"2 秒"、TTT3R 的"20 FPS / 6 GB"等具体数字均未引用进正文。

## 编译命令与结果

最近一次（2026-05-13）：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex  build/main
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

结果：

- 13 页 PDF 正常输出至 `build/main.pdf`。
- 0 × undefined citation / undefined reference。
- 0 × LaTeX Error。
- 0 × `Overfull \hbox`。
- 7 × `Underfull \hbox` —— 均为 `tab:foundation` 和 `tab:testtime` 的 CJK 列宽伸缩，符合优化提示词中"少量表格断行可接受"的容忍。
- MiKTeX 在 Windows 11 Home China 10.0.29585 上仍打印 "running on an unsupported version of Windows"。PDF 正常生成，记录为环境警告，不视为稿件失败。

## 未完成任务（按优先级）

1. **逐篇核对核心论文实验表**：尤其 MV-DUSt3R+、VGGT、Fast3R 之间的可对照基准；TTT3R 的硬件/吞吐声称。完成后可把若干处"尚需确认"升级为具体引用。
2. **2026 预印本代码与许可证核对**：LoGeR、Mem3R、OVGGT、PAS3R、FILT3R、RayMap3R、LongStream 的官方仓库与协议；目前正文用语已保守，无需立即修正，但发表前应一次性核对。
3. **图 1 可读性回访**：长序列分支当前列出 8 个模型，未来若读者反馈密集，可裁剪到代表 3–4 个 + "等"。
4. **样式备份**：如计划改投，提前确认 `unsrtnat` 与 `ctex` 是否在目标期刊/会议模板兼容。

## 风险与注意事项

- D²USt3R 标题在 `references.bib` 中使用 `\textsuperscript{2}`；当前 `unsrtnat` 风格下渲染正常，更换风格时应复检。
- `MiKTeX` 在当前 Windows 系统会持续打印 "unsupported Windows" 警告，但不影响输出。如换为 TeXLive 可消除该警告。
- `papers/`、`build/`、`guidance_skills/` 仍按 `.gitignore` 排除；不要无意中把生成 PDF 或第三方 PDF 提交进版本控制。
- 原 Typst 稿（`main.typ`、`review-template.typ`、`bib.yaml` 等）保留为历史快照，不再维护；如未来仍需 Typst 版本，应在 handoff 中重新声明。

## 最后更新时间

2026-05-13（绝对日期；操作日内）。
