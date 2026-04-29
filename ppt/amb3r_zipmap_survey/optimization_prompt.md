# 展示页优化用提示词

你是一名熟悉计算机视觉、3D 重建和学术汇报设计的研究助理。请基于当前 HTML 展示页，对「AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend」和「ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training」做一次内容和表达优化。

目标：

1. 保持页面是中文学术汇报风格，适合研究生课堂/组会展示。
2. 不要把 AMB3R 和 ZipMap 过度强行关联；两者只在背景和最后简短对照中连接。
3. 优先提升讲解清晰度：每页只讲一个核心观点，减少堆砌。
4. 保留技术细节：AMB3R 的 sparse voxel backend、metric-scale head、zero-conv 注入、VO/SfM pipeline；ZipMap 的 TTT fast weights、linear-time backbone、scene-state query、streaming variant。
5. 保留关键数字：AMB3R 7 tasks/13 datasets、约 50-80 H100 hours、VO 4.2 FPS、TUM/ETH3D ATE、SfM 指标；ZipMap 700+/750 frames、<10s、75 FPS、20x+ faster than VGGT、29 datasets、64 H100 training。
6. 明确局限：AMB3R 的局部 backend、二次复杂度、动态场景、VO drift、SfM false clusters；ZipMap 的超长场景分布外退化、NVS 高频模糊、短序列常数开销。
7. 页面视觉应克制、专业、信息密度适中；不要用营销式 hero、不要用渐变大色块堆叠、不要让正文溢出卡片。
8. 保持单文件 HTML，不依赖外部 JS/CSS。可以使用内联 CSS 和少量原生 JS 实现键盘翻页。

请输出：

1. 优化后的完整 HTML。
2. 修改摘要：列出你重排了哪些页面、删减或加强了哪些论点。
3. 三条讲者备注：分别解释 AMB3R、ZipMap、最后对照页应该如何口头展开。

内容准确性要求：

- 引用事实必须来自论文、项目页或官方 GitHub README。
- 不要编造未报告的 benchmark 数字。
- 对论文声称要用“作者报告/论文报告”表述，避免当成已被独立复现的事实。
- 如果页面内数字来自不同表格或不同协议，必须说明任务和数据集，不要混成一个总排名。
