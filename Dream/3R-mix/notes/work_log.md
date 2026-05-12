# Work log for 3R survey

Append-only log of substantive editing passes. Replaces verbal handoff for fine-grained history; `NEW_CHAT_HANDOFF.md` carries the latest top-level status.

## 2026-05-13 — Comprehensive optimization pass (LaTeX)

- **Files edited**: `main.tex`.
- **Files written**: `notes/work_log.md` (new), `NEW_CHAT_HANDOFF.md` (rewritten for LaTeX track).
- **Drivers**: `COMPREHENSIVE_OPTIMIZATION_PROMPT.md` (sections 1–12); `notes/fact_cards.md` and `notes/review_quality_audit.md` used as evidence-discipline references.

### Structural changes

- Author removed: `\author{KYKT Dream 调研组}` → `\author{}` to drop the internal project name from the title block. No other KYKT/Dream/Dream3R/agent/skill/workflow strings remain in `main.tex` (`Grep` verified).
- Section list aligned to the prompt's 10-section plan:
  1. 引言
  2. 从传统几何流程到点图表示
  3. 基础谱系：DUSt3R、MASt3R 与 SfM 接口  *(split from old §3.1)*
  4. 多视角规模化与统一视觉几何  *(split from old §3.2; now a section, not a subsection)*
  5. 视频、动态场景与 4D 重建
  6. 长序列重建中的状态、记忆与缓存
  7. 测试时验证、修正与先验输入  *(rewritten as three paragraphs covering Test3R/TTT3R/G-CUT3R/Pow3R/MASt3R-SfM and external priors)*
  8. 从几何预测到可查看输出
  9. 应用证据、复现边界与失败样本记录  *(new dedicated section; absorbs the application figure and matrix table)*
  10. 开放问题与结论  *(merged previous "讨论" four-point structure into the conclusion, dropping the standalone "结论" section)*

### Prose changes

- Abstract reworded to lead with the problem-branch organization principle (rather than the bare model list) and to make the three-tier evidence labeling (paper / official-code / local-smoke) explicit.
- Intro evidence-boundary paragraph rewritten to drop "本地流程跑通...质量领先" phrasing in favor of "可运行的本地复现只说明接口和依赖成立，不说明几何质量更优".
- Dynamic section: removed "通常被视为...重要代表" and "应更克制地理解"; replaced with direct mechanism descriptions for MonST3R/POMATO/D²USt3R/Easi3R/RayMap3R; added explicit "4D pointmap ≠ 4DGS asset" closing sentence.
- Long-sequence section: replaced the run-on model-listing paragraph with a four-class breakdown (spatial-pointer / causal-autoregressive / hybrid memory / budget-and-filter), each with one-sentence mechanism description and citation cluster.
- Test-time section: introduced three paragraphs covering (a) Test3R vs TTT3R consistency-vs-state distinction, (b) G-CUT3R/Pow3R/MASt3R-SfM differing prior-entry positions, (c) external auxiliary priors (Depth Pro / Metric3Dv2 / DINO / CoTracker / SpatialTracker / SAM2) framed as system-layer signals subject to prior conflicts.
- Output section: tightened Splatt3R/InstantSplat/NoPoSplat distinctions (uncalibrated pair vs. dense-stereo+GBA vs. canonical-frame unposed) and the rendering-vs-geometry caveat.
- New §9 prose explicitly defines the four evidence tiers (paper-proven, official-code, local-smoke-test, application-validated) plus "尚需确认", and ties them to figure / matrix table.
- Open-problems-and-conclusion section consolidates the previous four discussion points into one flowing paragraph plus a closing summary; both pieces avoid superlatives ("最强 / 领先 / 突破" — `Grep`-verified absent).

### New artifact

- Table 4: `tab:testtime` — "测试时机制与先验输入的进入位置和证据边界". Columns: 方法或先验 / 进入位置 / 修正约束信号 / 证据边界与风险. Rows cover Test3R, TTT3R, G-CUT3R, Pow3R, MASt3R-SfM, depth priors group, DINO group, tracking/segmentation group.

### Figure and table changes (summary)

- `fig:lineage` (Fig. 1) unchanged: DUSt3R-root taxonomy with six branches.
- `fig:memory` (Fig. 2 → still Fig. 2 by appearance order; in the new structure it appears in §6) unchanged: recurrent / spatial / hybrid / cache-policy quartet.
- `fig:application` (Fig. 4 by appearance) moved from §8 to new §9.
- `tab:foundation` rows tightened (replaced "领先" / "SOTA" wording with "更优" / "具体优劣需按实验表引用").
- `tab:dynamic` unchanged.
- `tab:memory` unchanged.
- `tab:testtime` added (new).
- `tab:application` moved to §9; "领先" cell text softened to "重建质量更优".

### Citation hygiene

- Removed `\nocite{*}`. All 43 bib keys in `references.bib` are now explicitly cited at least once (verified by comparing `grep` of `\citep{...}` against `@misc{...,}` entries).
- BibTeX style: `unsrtnat` (preserved).

### Build outcome

- Pipeline (manual, no `latexmk`):
  - `xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex`
  - `bibtex build/main`
  - `xelatex` × 2
- Final PDF: `build/main.pdf`, 13 pages, ~240 KB.
- Log status:
  - No `LaTeX Error`.
  - No `Undefined citation` or `Undefined reference`.
  - No `Overfull \hbox`.
  - 7 × `Underfull \hbox` warnings — all CJK column-wrap inside `tab:foundation` (line 130) and `tab:testtime` (line 233); accepted per the prompt's stated tolerance for table-induced underfulls.
- Environment noise: MiKTeX prints "running on an unsupported version of Windows" at every xelatex/bibtex invocation (Windows 11 Home China 10.0.29585). PDF generation is unaffected; recorded as environment warning, not a manuscript issue.

### Next-step candidates

- Per-paper benchmark reading: `notes/fact_cards.md` still flags benchmark numbers, license terms, and exact training-data details as "尚需确认". A targeted reading of 5–8 core PDFs would let us upgrade specific claims (e.g., MV-DUSt3R+ "2 seconds" hardware context, VGGT vs Fast3R quality comparison, TTT3R 20 FPS / 6 GB claims).
- 2026 preprints (LoGeR, Mem3R, OVGGT, PAS3R, FILT3R, RayMap3R, LongStream): code/checkpoint status not independently verified. Sentence-level claims are already kept neutral, but a targeted repository check before any external sharing remains warranted.
- D²USt3R bibliographic title contains `\textsuperscript{2}`; renders correctly under `unsrtnat`, but if the bib style is later changed (`acm`, `ieeetran`, etc.), revisit.
- Figure 1 (`fig:lineage`) currently lists all members per branch; once the survey is read by a third reviewer, consider compressing the long memory-branch list into a representative subset to avoid label crowding.

## 2026-05-13 — Figure optimization round (LaTeX)

- **Files edited**: `main.tex` (added `fig:paradigm` + small `fig:lineage` label tweak); `notes/figure_prompts.md` (full rewrite for current LaTeX state); `NEW_CHAT_HANDOFF.md` (figure-optimization subsection + updated未完成任务 + bumped timestamp).
- **New artifact**: `fig:paradigm` in §2 — TikZ two-row pipeline diagram. Top row: 特征提取与匹配 → 相机位姿估计 → 稀疏三角化 → 稠密 MVS/深度融合 → 全局 BA. Bottom row: 图像/图像对 → pointmap/深度/相机/置信度 → 对齐与一致性检查 → 点云/相机/匹配/Gaussian. Reused existing `arrows.meta`, `positioning`, `calc` libraries; no new package needed. Position: between the "三个直接后果" paragraph and the lineage paragraph; the consequences paragraph now opens with `这种变化的整体对照见图\,\ref{fig:paradigm}`.
- **Figure tweak**: `fig:lineage` streaming-branch label wraps to two lines (4+4 models) with trailing `等` to acknowledge §6 includes more models (PAS3R/FILT3R/LongStream) than fit the diagram.
- **Paper-figure embedding deferred**: `figures/` retains the DUSt3R/VGGT/MonST3R/CUT3R Fig.1 crops from the previous attempt, plus raw page-1 rasters. They are **not** referenced from `main.tex`. Reason: (1) prior crops were mixed quality — VGGT captured the title block, DUSt3R included caption tail, MonST3R/CUT3R were never visually verified; (2) per-paper reuse licenses for these arXiv preprints have not been independently confirmed; (3) the optimization prompt §0.4 defaults to "no embedded paper screenshots unless license confirmed". `notes/figure_prompts.md` records candidate re-crop coordinates (`crops_v2`) for any future attempt.
- **Compile**:
  - Same four-step pipeline as 2026-05-13 (xelatex / bibtex / xelatex / xelatex).
  - `build/main.pdf` is still 13 pages — `fig:paradigm` fit without page bloat.
  - Log: 0 errors, 0 undefined references/citations, 0 Overfull, 7 Underfull (same `tab:testtime` CJK column wraps as before; no new warning).
  - MiKTeX "unsupported Windows" message persists; PDF unaffected.
- **Caveat to user**: the chosen scope was "both new paradigm figure and embed paper figures"; only the first half landed in `main.tex`. The second half is staged in `figures/` but deferred pending license confirmation and crop re-verification. To proceed, follow `notes/figure_prompts.md` § "Paper-figure cache".

## 2026-05-13 — Paper-figure re-crop pass (not embedded)

- **Files edited**: none in `main.tex` this round. Only re-ran the PIL crop on the four cached page-1 rasters and overwrote `figures/<name>_fig1.png`.
- **Files written**: this `work_log.md` entry; `NEW_CHAT_HANDOFF.md` (timestamp + figure-cache subsection refreshed); `notes/figure_prompts.md` (cache table refreshed with new dimensions and visual-verification status).
- **Driver**: previous chat's handoff item "论文 Fig.1 嵌入决策" — re-crop with `crops_v2` coordinates, verify visually, then decide whether to embed.

### What changed on disk

- Re-applied `crops_v2` (plus minor manual tweaks) to the four `figures/<name>_p1-01.png` rasters. New dimensions on disk:
  - `dust3r_fig1.png` — 1365 × 630
  - `vggt_fig1.png` — 1380 × 465
  - `monst3r_fig1.png` — 1310 × 435
  - `cut3r_fig1.png` — 1385 × 360
- Raw page-1 rasters (`*_p1-01.png`, 1530 × 1980) are unchanged and remain alongside.

### Visual verification status

Read each cropped PNG one at a time (the previous attempt at multi-image Read hit a 32 MB request cap):

- `dust3r_fig1.png` — verified clean. Captures DUSt3R Fig.1 with pointmap outputs and rendered point clouds; no caption tail; no title-block fragment.
- `vggt_fig1.png` — verified clean. Captures the house/garden reconstruction with camera frusta and depth map; prior title-block issue is resolved.
- `monst3r_fig1.png` — verified clean. Captures the video-input strip, dynamic point cloud, and output labels (Video Depth / Camera Intrinsics / Dynamic & Static Masks).
- `cut3r_fig1.png` — **not yet visually verified**. The Read call returned `(media removed — rejected by API)` despite the file being only 623 KB on disk; cause unclear. Dimensions on disk look plausible (1385 × 360 — wide-and-short, consistent with a horizontal pipeline figure). Re-verification deferred.

### Not done this round (intentional)

- No `\includegraphics` blocks inserted into `main.tex`. Per `notes/figure_prompts.md` § "Figure policy", embedding is gated on independent license/reuse confirmation per paper. That check has not been performed.
- No xelatex/bibtex recompile. `build/main.pdf` is unchanged from the 2026-05-13 figure-optimization round (still 13 pages, same TikZ-only figure set).
- CUT3R crop visual confirmation still pending; if a future round wants to embed it, redo the Read in a clean context or open it in a viewer.

### Next-step candidates (unchanged in priority from prior entry)

1. Per-paper license/reuse confirmation for DUSt3R / VGGT / MonST3R / CUT3R before any `\includegraphics` embedding.
2. Visually re-verify `cut3r_fig1.png`.
3. If license cleared, insert four `\figure` blocks with attribution captions: 「图 X. 摘自 \citet{KEY}，仅用于综述说明，版权归原作者。」 — likely positions: DUSt3R/VGGT in §3–§4, MonST3R in §5, CUT3R in §6. Then xelatex × 3 + bibtex.
