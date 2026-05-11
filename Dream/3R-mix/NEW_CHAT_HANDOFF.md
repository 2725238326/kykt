# New chat handoff for 3R survey generation

Copy the following text into a new conversation when starting the full survey
writing pass.

```text
我们要继续写一篇中文 3R 模型综述，工作目录是 E:\kykt\Dream\3R-mix。请先读取 E:\kykt\Dream\3R-mix\GENERATION_PROMPT.md，并严格按里面的写作标准和目录推进。Typst 模板已经拉在 E:\kykt\Dream\3R-mix\simple-typst-thesis，当前综述骨架在 E:\kykt\Dream\3R-mix\main.typ，starter bibliography 在 E:\kykt\Dream\3R-mix\bib.yaml。

核心要求：
1. 用 Typst 写综述，不用 LaTeX。
2. 中文学术综述风格，语言自然正式，严肃剔除 AI 味、宣传腔和口号式表达。
3. 综述主题是近期 3R / feed-forward 3D reconstruction 模型谱系，不要把 Dream3R 写成中心；Dream 的长期调研经验只作为方法论和应用展望背景。
4. 覆盖 DUSt3R、MASt3R、MASt3R-SfM、Fast3R、MV-DUSt3R+、VGGT、Align3R、Pow3R、CUT3R、Spann3R、Point3R、STream3R、LONG3R、LoGeR、Mem3R、OVGGT、MonST3R、POMATO、D^2USt3R、Easi3R、RayMap3R、Test3R、TTT3R、G-CUT3R、Splatt3R、InstantSplat、NoPoSplat、3DGS/4DGS，以及必要的 Depth Anything、DINO、CoTracker、SAM2 等支撑先验。
5. 先建 notes/model_inventory.md、notes/paper_inventory.md、notes/figure_prompts.md，再扩充 bib.yaml 和 main.typ。
6. 本地没有的论文 PDF 可以从 arXiv、CVF、ECCV、项目官网或 GitHub 官方链接下载到 E:\kykt\Dream\3R-mix\papers，并在 paper_inventory 记录来源。
7. 图可以用本地论文图、Typst 绘制、Mermaid 草图或 AI 生成；AI 生成图的提示词写入 notes/figure_prompts.md。
8. 所有性能、SOTA、代码可用性、应用可行性判断都要有来源或标注“尚需确认”。不要把流程跑通写成质量领先。

先做第一阶段：整理模型清单、论文清单、章节结构和图示计划，然后再开始写 main.typ 正文。
```
