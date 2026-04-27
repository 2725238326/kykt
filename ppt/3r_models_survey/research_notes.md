# 3R 系列模型综述笔记

本目录用于整理一版新的 3R 系列模型综述。当前页面不是最终汇报版，而是内容骨架和讲解路线。

## 讲解主线

传统 3D reconstruction 的流程通常是：

1. 特征提取与匹配；
2. 相机内外参或相对位姿估计；
3. 三角化；
4. 多视图深度融合；
5. 全局优化和点云/网格重建。

DUSt3R 系列的核心变化是：把很多中间几何变量收进一个可学习的 dense 3D 表示里。它不先要求完整 camera calibration 或 pose，而是直接从图像预测 pointmap，然后从 pointmap 导出深度、匹配、位姿和重建结果。

## 模型关系

| 模型 | 位置 | 讲解重点 |
|---|---|---|
| DUSt3R | 基座 | pointmap regression，统一深度、位姿、匹配、重建 |
| MASt3R | 匹配增强 | 在 DUSt3R 基础上增加 dense matching / descriptor 能力 |
| Spann3R | 全局记忆 | 用 spatial memory 缓解 pairwise pointmap 后续全局对齐 |
| MonST3R | 动态场景 | 把 DUSt3R 的几何优先范式推进到 dynamic scenes / 4D reconstruction |
| Align3R | 视频深度 | 用 DUSt3R 对齐单目深度，获得时间一致深度和相机位姿 |
| Pow3R | 先验利用 | 允许推理时利用相机、位姿、深度等可选先验 |
| VGGT | 参考方向 | 更激进的端到端视觉几何模型，直接输出相机、深度、pointmap、tracks |

## 和教师学生 / Depth Anything 的衔接

我们前面讨论的核心点：

- Depth Anything 的学生最终输出深度图，不输出语义标签。
- 但学生中间特征会吸收语义结构，尤其通过 DINOv2 feature alignment。
- `L_feat` 的作用不是让学生生成语义，而是让学生使用语义结构来预测深度。
- Align3R 让我们看到深度估计在视频几何一致性中的作用。
- Depth Anything 提供单帧高质量 depth prior，Align3R/DUSt3R 负责跨帧和跨视角对齐。

可用于汇报的一句话：

> 3R 系列负责把图像组织成一致的 3D 表示；教师学生模型负责把难获得的深度监督扩展到海量数据。两者在单目视频几何理解中自然接上。

## 参考入口

- DUSt3R: https://arxiv.org/abs/2312.14132
- CVPR 2024 open access: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html
- MASt3R: https://arxiv.org/abs/2406.09756
- ECCV 2024 poster: https://eccv.ecva.net/virtual/2024/poster/523
- Spann3R: https://arxiv.org/abs/2408.16061
- MonST3R: https://arxiv.org/abs/2410.03825
- Align3R: https://arxiv.org/abs/2412.03079
- Align3R CVPR 2025 PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_Align3R_Aligned_Monocular_Depth_Estimation_for_Dynamic_Videos_CVPR_2025_paper.pdf
- Pow3R: https://arxiv.org/abs/2503.17316
- VGGT: https://arxiv.org/abs/2503.11651
