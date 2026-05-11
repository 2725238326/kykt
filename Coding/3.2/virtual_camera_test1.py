from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
SENSOR_WIDTH_MM = 32
OUTPUT_DIR = Path(__file__).with_name("virtual_camera_results")

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def get_cube_points():
    """
    生成一个简单的 3D 立方体顶点坐标，用于测试。
    立方体中心在原点 (0,0,0)，边长为 2。
    """
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    ).T
    return points


def get_camera_intrinsics(image_width, image_height, f_mm, sensor_w_mm):
    """
    计算内参矩阵 K。

    f_mm: 焦距（毫米）
    sensor_w_mm: 传感器宽度（毫米）
    """
    # 1. 计算像素焦距。这里假设像素是正方形，因此 fx = fy。
    f_pixels = (f_mm / sensor_w_mm) * image_width

    # 2. 主点通常取图像中心。
    cx = image_width / 2.0
    cy = image_height / 2.0

    # 3. 构造 pinhole camera 的 3x3 内参矩阵。
    k_matrix = np.array(
        [
            [f_pixels, 0.0, cx],
            [0.0, f_pixels, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return k_matrix


def rotation_y(degrees):
    """构造绕 Y 轴旋转的 3x3 矩阵。"""
    theta = np.deg2rad(degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array(
        [
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t],
        ],
        dtype=float,
    )


def project_points(points_world, k_matrix, r_matrix, t_vector):
    """
    核心函数：将 3D 世界坐标投影到 2D 像素坐标。

    参数:
    points_world: (3, N) 3D 点集
    k_matrix: (3, 3) 内参矩阵
    r_matrix: (3, 3) 旋转矩阵 (World -> Camera)
    t_vector: (3, 1) 平移向量 (World -> Camera)

    返回:
    points_pixel: (2, N) 2D 像素坐标
    """
    # 步骤 1: 世界坐标系 -> 相机坐标系，P_cam = R * P_world + t。
    points_cam = r_matrix @ points_world + t_vector

    # 在真实 SfM 中，这步用于剔除错误的解。
    invalid_depths = points_cam[2, :] <= 0
    if np.any(invalid_depths):
        print(f"警告: 有 {int(np.sum(invalid_depths))} 个点位于相机背面或成像平面上！")

    # 步骤 2: 相机坐标系 -> 齐次图像坐标。
    points_img_homo = k_matrix @ points_cam

    # 步骤 3: 透视除法，得到像素坐标。
    u = points_img_homo[0, :] / points_img_homo[2, :]
    v = points_img_homo[1, :] / points_img_homo[2, :]
    return np.vstack((u, v))


def draw_camera_view(ax, points_2d, title, flip_y=True, show_labels=True):
    """绘制一个二维相机成像视图。"""
    y_values = points_2d[1, :]
    colors = ["#d7191c" if y < IMAGE_HEIGHT / 2 else "#2c7bb6" for y in y_values]

    ax.set_title(title)
    ax.scatter(points_2d[0, :], points_2d[1, :], c=colors, marker="o", s=46, zorder=3)
    for p1, p2 in EDGES:
        ax.plot(
            [points_2d[0, p1], points_2d[0, p2]],
            [points_2d[1, p1], points_2d[1, p2]],
            color="#1f4e79",
            linewidth=1.6,
        )

    if show_labels:
        for index, (u_value, v_value) in enumerate(points_2d.T):
            ax.annotate(
                str(index),
                (u_value, v_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )

    ax.set_xlim(0, IMAGE_WIDTH)
    if flip_y:
        ax.set_ylim(IMAGE_HEIGHT, 0)
    else:
        ax.set_ylim(0, IMAGE_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("u / pixel")
    ax.set_ylabel("v / pixel")
    ax.grid(True, linewidth=0.5, alpha=0.45)


def draw_world_view(ax, points_3d, title, camera_position=(0.0, 0.0, -5.0)):
    """绘制三维世界坐标示意图。"""
    ax.set_title(title)
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c="#1f77b4")
    for p1, p2 in EDGES:
        ax.plot(
            [points_3d[0, p1], points_3d[0, p2]],
            [points_3d[1, p1], points_3d[1, p2]],
            [points_3d[2, p1], points_3d[2, p2]],
            color="#1f77b4",
            linewidth=1.2,
        )
    ax.scatter(*camera_position, c="black", marker="^", s=120, label="Camera")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-6, 3)
    ax.view_init(elev=24, azim=-62)


def save_baseline(points_3d):
    """保存基准相机投影图。"""
    k_matrix = get_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, 35, SENSOR_WIDTH_MM)
    r_matrix = np.eye(3)
    t_vector = np.array([[0.0], [0.0], [5.0]])
    points_2d = project_points(points_3d, k_matrix, r_matrix, t_vector)

    fig = plt.figure(figsize=(10.5, 5.0), dpi=150)
    ax_2d = fig.add_subplot(1, 2, 1)
    draw_camera_view(ax_2d, points_2d, "Baseline: f=35mm, t=[0,0,5]")
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    draw_world_view(ax_3d, points_3d, "World View")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "baseline_original.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_comparison(case_name, panels, filename, figsize=(11.0, 4.2)):
    """保存多个相机视图组成的对照图。"""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, dpi=150)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        draw_camera_view(
            ax,
            panel["points_2d"],
            panel["title"],
            flip_y=panel.get("flip_y", True),
            show_labels=panel.get("show_labels", True),
        )

    fig.suptitle(case_name, fontsize=13)
    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_experiments():
    """运行四个问题对应的实验，并保存所有图像结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points_3d = get_cube_points()
    base_r = np.eye(3)
    base_t = np.array([[0.0], [0.0], [5.0]])
    base_k = get_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, 35, SENSOR_WIDTH_MM)
    base_points = project_points(points_3d, base_k, base_r, base_t)

    outputs = {}
    outputs["baseline"] = save_baseline(points_3d)

    q1_k = get_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, 100, SENSOR_WIDTH_MM)
    q1_points = project_points(points_3d, q1_k, base_r, base_t)
    outputs["q1"] = save_comparison(
        "Q1: focal length comparison",
        [
            {"points_2d": base_points, "title": "f=35mm"},
            {"points_2d": q1_points, "title": "f=100mm"},
        ],
        "q1_focal_length_compare.png",
    )

    q2_t = np.array([[1.0], [0.0], [5.0]])
    q2_points = project_points(points_3d, base_k, base_r, q2_t)
    outputs["q2"] = save_comparison(
        "Q2: camera translation comparison",
        [
            {"points_2d": base_points, "title": "t=[0,0,5]"},
            {"points_2d": q2_points, "title": "t=[1,0,5]"},
        ],
        "q2_translation_compare.png",
    )

    outputs["q3"] = save_comparison(
        "Q3: image coordinate y-axis comparison",
        [
            {"points_2d": base_points, "title": "With plt.ylim(H, 0)", "flip_y": True},
            {"points_2d": base_points, "title": "Without y-axis flip", "flip_y": False},
        ],
        "q3_y_axis_flip_compare.png",
    )

    q4_r = rotation_y(45)
    q4_points = project_points(points_3d, base_k, q4_r, base_t)
    outputs["q4"] = save_comparison(
        "Q4: 45-degree rotation around Y-axis",
        [
            {"points_2d": base_points, "title": "R = I"},
            {"points_2d": q4_points, "title": "R = Ry(45 deg)"},
        ],
        "q4_y_rotation_45.png",
    )

    return outputs


def main():
    k_matrix = get_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, 35, SENSOR_WIDTH_MM)
    print("内参矩阵 K:")
    print(k_matrix)

    outputs = run_experiments()
    print("\n实验图片已保存:")
    for key, path in outputs.items():
        print(f"{key}: {path}")

    print("\n四个问题的结论:")
    print("Q1: 焦距从 35mm 增加到 100mm 后，像素焦距变大，立方体在图像上变大。")
    print("Q2: t=[1,0,5] 表示世界点在相机坐标中 X 增加，相当于相机向左平移，图像中立方体向右移动。")
    print("Q3: 如果不执行 plt.ylim(H,0)，显示结果会上下翻转；图像坐标系 v 轴向下，而笛卡尔坐标系 y 轴向上。")
    print("Q4: 绕 Y 轴 45 度旋转可用 Ry 矩阵实现，投影会出现明显的水平透视倾斜和深度变化。")


if __name__ == "__main__":
    main()
