# 3R survey generation prompt

Use this prompt for the agent that will write the Typst survey in
`E:\kykt\Dream\3R-mix`.

```text
你将撰写一篇中文 3R 模型综述，工作目录是 E:\kykt\Dream\3R-mix。请使用 Typst，模板使用该目录下的 simple-typst-thesis。正文从 main.typ 开始，必要时可以拆分到 src/。不要使用 LaTeX，不要改到 Dream 其他目录，除非只是读取资料。

写作目标：
写一篇关于近期 3R / feed-forward 3D reconstruction 模型谱系的综述。综述要覆盖 DUSt3R 及其后续相关方法，也要结合 Dream 长期调研中积累的判断：为什么要做这个方向、各方法解决什么问题、局限在哪里、如何走向真实应用和成果产出。

语言要求：
1. 使用正常中文学术论文风格，严肃、自然、克制。
2. 严禁宣传腔、口号式总结和明显 AI 风格套话。
3. 不要频繁使用“可控、可验证、可扩展”“颠覆”“突破”“空间智能”等泛泛表达。
4. 技术名可以保留英文，但解释和论证要用中文写清楚。
5. 不要把 Dream3R 写成综述中心。Dream 的经验可以放在方法论或应用展望中，作为调研和原型经验，而不是替代文献综述。
6. 任何性能、SOTA、代码可用性和应用可行性判断都要标明依据。没有确认的地方写“尚需确认”或“需要进一步验证”。

必须读取的本地资料：
- E:\kykt\Dream\TASK_SNAPSHOT.md
- E:\kykt\Dream\code\dream3r\SOTA_FEATURE_MATRIX.md
- E:\kykt\Dream\code\dream3r\RESEARCH_LITERATURE_MAP.md
- E:\kykt\Dream\code\dream3r\RESEARCH_BASE_AND_INNOVATIONS.md
- E:\kykt\Dream\code\dream3r\NEXT_PHASE_ROADMAP.md
- E:\kykt\Dream\sources\FRONTIER_SOURCE_MAP.md
- E:\kykt\Dream\registry\source_registry.md
- E:\kykt\Dream\units\REPRODUCTION_READINESS_MATRIX.md
- E:\kykt\Dream\units\RESEARCH_UNIT_BANK.md
- E:\kykt\Dream\literature\INDEX.md
- E:\kykt\Dream\literature\SPINE_CRITIC.md
- E:\kykt\Dream\literature\SPINE_MEMORY.md
- E:\kykt\Dream\literature\SPINE_PERMANENCE.md
- E:\kykt\Dream\literature\SPINE_COMPOSER.md
- E:\kykt\ppt\3r_models_survey\research_notes.md

优先覆盖的模型和方向：
1. 基础点图/匹配线：DUSt3R、MASt3R、MASt3R-SfM。
2. 多视角和规模化：Fast3R、MV-DUSt3R+、VGGT。
3. 视频深度和先验利用：Align3R、Pow3R、Depth Anything 相关深度先验。
4. 长序列和记忆：CUT3R、Spann3R、Point3R、STream3R、LONG3R、LoGeR、Mem3R、OVGGT。
5. 动态和 4D：MonST3R、POMATO、D^2USt3R、Easi3R、RayMap3R。
6. 测试时验证和更新：Test3R、TTT3R、G-CUT3R。
7. 可视化输出和应用：Splatt3R、InstantSplat、NoPoSplat、3D Gaussian Splatting、4DGS 相关方法。
8. 支撑先验：DINOv2/v3、CoTracker、SAM2、SpatialTracker、Depth Pro、Metric3D v2 等，只作为辅助线索，不要喧宾夺主。

建议章节结构：
1. 引言：3R 方法的范式变化和综述范围。
2. 从传统几何流程到 pointmap 表示。
3. DUSt3R 及匹配增强分支。
4. 多视角规模化与统一视觉几何模型。
5. 视频、动态场景和 4D 重建。
6. 长序列重建中的状态、记忆和缓存机制。
7. 测试时验证、修正和自适应。
8. 面向应用的输出表示：点云、深度、confidence、mesh、Gaussian。
9. 方法比较表：输入假设、输出表示、是否需要相机、是否支持动态、是否支持长序列、工程可复现性、应用落地难度。
10. 图示和应用路径：如何从图像输入走向可查看的重建结果、报告和系统集成。
11. 经验总结与开放问题。

必须生成或预留的图：
1. 3R 模型谱系图：DUSt3R -> MASt3R / Fast3R / Spann3R / MonST3R / CUT3R / VGGT 等。
2. 能力分类图：匹配、规模化、多视角、长序列、动态场景、验证修正、Gaussian 输出。
3. 应用落地图：输入图像/视频 -> 几何预测 -> 质量检查 -> 结果表示 -> 可视化/报告/KYKT 集成。
图可以先用 Typst 原生图、Mermaid 风格草图或 AI 生成图占位。若使用 AI 生成图，提示词必须写入 notes/figure_prompts.md。

资料获取规则：
1. 优先使用本地文档中已有链接。
2. 本地没有 PDF 时，从 arXiv、CVF、ECCV、项目官网或 GitHub 官方链接下载到 papers/。
3. 下载后在 notes/paper_inventory.md 记录来源、文件名、是否已读、是否已引用。
4. 不引用无法确认来源的二手描述。

写作推进顺序：
1. 先建立 notes/model_inventory.md，列出模型、年份、输入输出、核心机制、局限、应用状态。
2. 扩充 bib.yaml，不要只引用 URL；能确认作者和会议时补齐。
3. 写 main.typ 的完整一级和二级标题。
4. 每个章节先写 2-4 段扎实正文，再补表格和图。
5. 最后统一语言，删除 AI 风格表达，检查是否有过度声称。

交付要求：
- main.typ 可以编译。
- bib.yaml 至少覆盖核心模型。
- notes/model_inventory.md、notes/paper_inventory.md、notes/figure_prompts.md 存在。
- 综述语言自然正式，不像自动生成的宣传文案。
```
