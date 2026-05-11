#set page(
  paper: "a4",
  margin: (x: 2.4cm, y: 2.2cm),
)
#set text(
  font: ("Microsoft YaHei", "SimSun"),
  lang: "zh",
  size: 10.5pt,
)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#align(center)[
  #text(size: 18pt, weight: "bold")[虚拟相机投影实验报告]

  #v(0.7em)
  #grid(
    columns: (auto, auto),
    column-gutter: 2em,
    row-gutter: 0.4em,
    [课程：科研课堂], [姓名 / 学号：24374367-纪博闻],
    [实验日期：2026 年 3 月 2 日], 
  )
]

#v(1em)

= 实验目的

本次实验使用针孔相机模型，将一个三维立方体投影到二维图像平面。实验要求补全相机内参矩阵和透视除法，并在此基础上观察焦距、相机平移、图像坐标系方向以及相机旋转对成像结果的影响。

= 实验环境

实验在 conda虚拟环境中完成，使用 Python、NumPy 和 Matplotlib 编写并运行程序。

主要环境如下：

#table(
  columns: (35%, 65%),
  inset: 6pt,
  align: (left, left),
  [Python 环境], [`conda` 虚拟环境 `kykt`],
  [主要库], [`numpy 2.2.5`，`matplotlib 3.10.8`],

)

= 实验原理

针孔相机模型可分为两个步骤：先把世界坐标系中的点变换到相机坐标系，再把相机坐标投影到图像平面。

设三维点为 $P_w$，旋转矩阵为 $R$，平移向量为 $t$，则相机坐标为：

$ P_c = R P_w + t $

相机内参矩阵写作：

$ K = mat(
  f_x, 0, c_x;
  0, f_y, c_y;
  0, 0, 1
) $

其中 $f_x$、$f_y$ 为像素单位下的焦距，$c_x$、$c_y$ 为主点坐标。本实验图像分辨率为 $800 times 600$，默认焦距为 35 mm，传感器宽度为 32 mm，因此像素焦距为：

$ f_"px" = 35 / 32 times 800 = 875 $

默认内参矩阵为：

$ K = mat(
  875, 0, 400;
  0, 875, 300;
  0, 0, 1
) $

得到齐次图像坐标 $p' = K P_c = (x', y', z')^T$ 后，需要进行透视除法：

$ u = x' / z', quad v = y' / z' $

这一步将齐次坐标转换为实际的像素坐标，是三维点投影到二维图像中的关键步骤。

= 程序实现

程序首先生成以原点为中心、边长为 2 的立方体八个顶点。随后根据图像大小、焦距和传感器宽度构造内参矩阵 $K$，再使用给定的 $R$ 与 $t$ 完成世界坐标到相机坐标的变换。最后通过透视除法得到二维像素坐标，并绘制立方体的投影连线。

内参矩阵的实现如下：

```python
k_matrix = np.array([
    [f_pixels, 0.0, cx],
    [0.0, f_pixels, cy],
    [0.0, 0.0, 1.0],
])
```

投影与透视除法的实现如下：

```python
points_cam = r_matrix @ points_world + t_vector
points_img_homo = k_matrix @ points_cam
u = points_img_homo[0, :] / points_img_homo[2, :]
v = points_img_homo[1, :] / points_img_homo[2, :]
```

= 基准实验结果

基准实验采用默认参数：焦距为 35 mm，旋转矩阵为单位矩阵，平移向量为 $t = [0, 0, 5]^T$。此时相机正对立方体，投影结果基本保持对称。

#figure(
  image("virtual_camera_results/baseline_original.png", width: 95%),
  caption: [默认参数下的二维投影与三维场景示意图],
)

= 问题分析与实验结果

== Q1：焦距改变对投影大小的影响

题目要求将焦距从 35 mm 改为 100 mm，并判断图像中的立方体变大还是变小。

从投影公式 $u = f_x X/Z + c_x$ 和 $v = f_y Y/Z + c_y$ 可以看出，在三维点位置和深度不变的情况下，焦距越大，像素坐标偏离主点越远。因此焦距从 35 mm 增加到 100 mm 后，立方体在图像中会变大。

#figure(
  image("virtual_camera_results/q1_focal_length_compare.png", width: 95%),
  caption: [焦距 35 mm 与 100 mm 的投影对比],
)

实验结论：立方体变大。焦距增大相当于视场角变小，同一物体在图像中占据更大的范围。

== Q2：相机平移对投影位置的影响

题目要求将平移向量改为 $t = [1, 0, 5]^T$，并判断立方体在图像中的移动方向。

在程序中，世界点到相机坐标的变换为 $P_c = R P_w + t$。当 $t_x$ 由 0 变为 1 时，每个点在相机坐标系中的 $X$ 坐标增加。代入投影公式后，像素横坐标 $u$ 随之增大，所以立方体在图像中向右移动。

#figure(
  image("virtual_camera_results/q2_translation_compare.png", width: 95%),
  caption: [平移向量改变前后的投影对比],
)

实验结论：立方体向右移动。若从相机运动的角度理解，这相当于相机向左移动，因此图像中的物体向相反方向移动。

== Q3：图像坐标系与笛卡尔坐标系的差异

题目要求去掉 `plt.ylim(H, 0)`，观察图像显示是否上下翻转。

Matplotlib 默认采用数学坐标系，纵轴向上；而图像坐标系通常以左上角为原点，横轴向右，纵轴向下。因此在显示相机图像时，需要使用 `plt.ylim(H, 0)` 将纵轴方向反过来。若不这样处理，投影结果会按数学坐标系显示，看起来相对于图像坐标系上下颠倒。

#figure(
  image("virtual_camera_results/q3_y_axis_flip_compare.png", width: 95%),
  caption: [翻转与不翻转纵轴时的显示对比],
)

实验结论：不翻转纵轴时，图像相对于正常图像坐标会上下倒置。这说明图像坐标系的 $v$ 轴方向与数学笛卡尔坐标系的 $y$ 轴方向相反。

== Q4：绕 Y 轴旋转 45 度

题目要求构造绕 $Y$ 轴旋转 45 度的旋转矩阵。采用的矩阵为：

$ R_y(theta) = mat(
  cos theta, 0, sin theta;
  0, 1, 0;
  -sin theta, 0, cos theta
) $

当 $theta = 45 degree$ 时，代码实现如下：

```python
theta = np.deg2rad(45)
R = np.array([
    [np.cos(theta), 0.0, np.sin(theta)],
    [0.0, 1.0, 0.0],
    [-np.sin(theta), 0.0, np.cos(theta)],
])
```

旋转后，立方体各顶点在相机坐标系中的横向位置和深度都会改变，因此二维投影不再保持正对相机时的对称形态，而呈现出明显的透视变化。

#figure(
  image("virtual_camera_results/q4_y_rotation_45.png", width: 95%),
  caption: [单位旋转矩阵与绕 Y 轴旋转 45 度的投影对比],
)

实验结论：绕 $Y$ 轴旋转 45 度后，立方体投影出现水平方向的形变和透视变化。这说明相机外参中的旋转矩阵会直接改变三维点的相机坐标，从而影响最终成像。

= 实验总结

本次实验完成了虚拟相机投影程序的关键部分，并分别考察了四个参数或显示设置对结果的影响。实验结果表明，焦距决定投影尺度，平移会改变物体在图像中的位置，图像坐标系的纵轴方向与数学坐标系不同，而旋转矩阵会改变三维点在相机坐标系中的方向和深度。这些现象与针孔相机模型的数学表达是一致的。
