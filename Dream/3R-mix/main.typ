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
    近年来，基于图像的三维重建从传统的匹配、位姿估计和多视图融合流程，逐渐转向以可学习几何表示为核心的端到端或准端到端方法。DUSt3R 系列及其后续工作将 pointmap、dense matching、视频深度、动态场景、长序列记忆和 pose-free reconstruction 等方向连接在一起，形成了新的 3R 方法谱系。本文计划围绕模型关系、核心表示、适用场景、工程可复现性和应用落地路径，对这一系列模型进行综述，并总结本项目在调研、原型实现和实验规划中形成的经验。
  ],
)

= Introduction

近年来，3D reconstruction 相关工作出现了明显的范式变化。传统流程通常依赖特征匹配、相机位姿估计、三角化和多视图融合；近期 3R 模型则更多地把深度、匹配、位姿和点云表示纳入统一的可学习几何输出中。DUSt3R 是这一变化中的代表性工作，后续的 MASt3R、Fast3R、Spann3R、MonST3R、CUT3R、VGGT 等模型进一步扩展了匹配、速度、长序列、动态场景和统一视觉几何预测能力。

本文不把这些模型简单列成时间线，而是从问题设置和能力分化出发，讨论它们分别解决了哪些限制、引入了哪些新的系统代价，以及在真实应用中如何组合使用。

== Scope

本文重点覆盖以下方向：

- pointmap-based reconstruction and matching: DUSt3R, MASt3R, MASt3R-SfM;
- scalable and multi-view reconstruction: Fast3R, MV-DUSt3R+, VGGT;
- streaming and memory-based 3R: CUT3R, Spann3R, Point3R, STream3R, LONG3R, LoGeR, Mem3R;
- dynamic and 4D reconstruction: MonST3R, POMATO, D^2USt3R, Easi3R, RayMap3R;
- test-time verification and adaptation: Test3R, TTT3R, G-CUT3R;
- pose-free Gaussian and deployable 3D outputs: Splatt3R, InstantSplat, NoPoSplat, 3DGS-related methods;
- supporting priors and application links: Depth Anything, DINO, CoTracker, SAM-style masks, and KYKT-style result presentation.

== Writing principle

The review should distinguish three kinds of statements:

- paper-level claims reported by the original authors;
- implementation or reproduction observations from local work;
- design interpretations and open hypotheses.

= From classical reconstruction to learned pointmaps

TODO: explain the classical pipeline, why camera pose and matching are bottlenecks, and how DUSt3R reframes the output as pointmaps.

= Core branches of recent 3R models

== Reconstruction and matching

TODO: DUSt3R, MASt3R, MASt3R-SfM.

== Scaling to more views

TODO: Fast3R, MV-DUSt3R+, VGGT.

== Streaming state and spatial memory

TODO: CUT3R, Spann3R, Point3R, STream3R, LONG3R, LoGeR, Mem3R.

== Dynamic scenes and 4D reconstruction

TODO: MonST3R, POMATO, D^2USt3R, Easi3R, RayMap3R.

== Test-time verification and correction

TODO: Test3R, TTT3R, G-CUT3R, geometry critic patterns.

== Pose-free Gaussian and visual output

TODO: Splatt3R, InstantSplat, NoPoSplat, 3DGS output paths.

= Comparative taxonomy

TODO: build tables by input assumption, output representation, temporal handling,
memory mechanism, dynamic-scene support, model cost, and application readiness.

= Application and deployment considerations

TODO: discuss how reconstruction outputs become usable artifacts: depth, point
clouds, confidence maps, mesh/Gaussian outputs, visual inspection, report
generation, failure-region marking, and integration with KYKT-style workflows.

= Lessons from the Dream work

TODO: summarize methodological lessons from the long-running Dream research
process: do not choose a single model prematurely; use evidence labels; separate
flow validation from quality claims; design ablations before claiming
architecture benefit; plan figures and deliverables around what has actually
been verified.

= Conclusion

TODO: write after the model inventory and paper reading are complete.

#pagebreak(weak: true)
#set page(header: [])
= Bibliography
#bibliography("bib.yaml", style: "apa", title: none)
