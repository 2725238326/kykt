# 教师学生模型调研笔记

## 汇报主线

1. 教师学生模型不是单个算法，而是一种训练角色分工：teacher 负责生产更强、更贵、更结构化的监督信号，student 负责吸收并部署。
2. 经典知识蒸馏的核心是 soft target：相比 one-hot，它保留类别相似性和模型不确定性。
3. 半监督/自训练中的 teacher 负责给无标签数据生成 pseudo label，但风险是复制 teacher 的错误。
4. 特征蒸馏让 student 对齐中间表示、注意力或 embedding，不只对齐最终答案。
5. Depth Anything v1/v2 是很适合课堂讲解的案例：它把教师学生框架从“小模型压缩”扩展成“大规模数据引擎”。

## Depth Anything v1 要点

- 论文：Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data, CVPR 2024.
- 目标：构建鲁棒的单目深度估计 foundation model。
- 数据：1.5M 有标签图像训练初始 teacher，62M 无标签真实图像由 teacher 自动生成 dense pseudo depth。
- 已加入 deck 的原图素材：
  - `images/v1_table1_data.png`：论文 Table 1，训练数据来源。
  - `images/v1_fig2_pipeline.png`：论文 Figure 2，v1 训练 pipeline。
  - `images/v1_table2_zero_shot.png`：论文 Table 2，zero-shot 相对深度结果。
  - `images/v1_teaser_web.jpg`：官方 GitHub teaser，实际预测效果。
- 关键细节：直接混合伪标签不一定提升，因为 student 可能只复制 teacher 的输出和错误。
- 改进：对无标签图像施加强颜色扰动、模糊、CutMix 等强扰动，让 student 在更难输入上恢复 clean pseudo labels。
- 语义辅助：用冻结 DINOv2 encoder 的 feature alignment loss 帮助深度模型继承语义先验。

## Depth Anything v2 要点

- 论文：Depth Anything V2, NeurIPS 2024.
- 相比 v1 的三条实践：用合成图像替换所有 labeled real images；扩大 teacher 容量；通过大规模 pseudo-labeled real images 教 student。
- 数据：595K 精确合成图像 + 62M 伪标注真实图像。
- 逻辑：synthetic labels 更精确，但有域偏移和多样性不足；real unlabeled images 作为桥接数据，缓解纯合成训练和真实世界之间的分布差异。
- 模型尺度：官方摘要说明提供 25M 到 1.3B 参数多尺度模型。
- 已加入 deck 的原图素材：
  - `images/v2_fig5_6_synthetic_transfer.png`：论文 Figure 5 / 6，synthetic-to-real 迁移与失败案例。
  - `images/v2_fig7_pipeline.png`：论文 Figure 7，v2 三段式 pipeline。
  - `images/v2_fig8_9_da2k.png`：论文 Figure 8 / 9，真实 benchmark 噪声与 DA-2K。
  - `images/v2_teaser_web.jpg`：官方 GitHub teaser，质量、速度、参数量对比。

## 可引用来源

- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network: https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/
- Depth Anything v1 arXiv: https://arxiv.org/abs/2401.10891
- Depth Anything v1 full text: https://ar5iv.labs.arxiv.org/html/2401.10891v2
- Depth Anything v2 arXiv: https://arxiv.org/abs/2406.09414
- Depth Anything v2 full text: https://ar5iv.labs.arxiv.org/html/2406.09414v2
- Depth Anything V2 official GitHub: https://github.com/DepthAnything/Depth-Anything-V2
- DINOv2 arXiv: https://arxiv.org/abs/2304.07193
- DINO arXiv: https://arxiv.org/abs/2104.14294
