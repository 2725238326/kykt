import matplotlib.pyplot as plt
import numpy as np


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
    计算内参矩阵 K
    f_mm: 焦距（毫米）
    sensor_w_mm: 传感器宽度（毫米）
    """
    f_pixels = (f_mm / sensor_w_mm) * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0

    # [补全] 原始代码这里要求构造 3x3 的相机内参矩阵 K。
    k_matrix = np.array(
        [
            [f_pixels, 0.0, cx],
            [0.0, f_pixels, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return k_matrix


def project_points(points_world, k_matrix, r_matrix, t_vector):
    """
    核心函数：将 3D 世界坐标投影到 2D 像素坐标

    参数:
    points_world: (3, N) 3D 点集
    k_matrix: (3, 3) 内参矩阵
    r_matrix: (3, 3) 旋转矩阵
    t_vector: (3, 1) 平移向量

    返回:
    points_pixel: (2, N) 2D 像素坐标
    """
    points_cam = r_matrix @ points_world + t_vector

    negative_depths = points_cam[2, :] <= 0
    if np.any(negative_depths):
        count = int(np.sum(negative_depths))
        print(f"Warning: {count} point(s) are on or behind the camera plane.")

    points_img_homo = k_matrix @ points_cam

    # [补全] 原始代码这里要求完成透视除法：
    # u = x' / z', v = y' / z'
    u = points_img_homo[0, :] / points_img_homo[2, :]
    v = points_img_homo[1, :] / points_img_homo[2, :]

    return np.vstack((u, v))


def plot_projection(points_2d, points_3d, image_width, image_height):
    """可视化二维投影结果和三维场景。"""
    edges = [
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

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Camera View (2D Image)")
    plt.scatter(points_2d[0, :], points_2d[1, :], c="r", marker="o")
    for p1, p2 in edges:
        plt.plot(
            [points_2d[0, p1], points_2d[0, p2]],
            [points_2d[1, p1], points_2d[1, p2]],
            "b-",
        )
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.gca().set_aspect("equal")
    plt.grid(True)

    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.set_title("World View (3D Scene)")
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c="b")
    ax.scatter(0, 0, -5, c="k", marker="^", s=100, label="Camera")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def main():
    image_width, image_height = 800, 600
    k_matrix = get_camera_intrinsics(
        image_width=image_width,
        image_height=image_height,
        f_mm=35,
        sensor_w_mm=32,
    )
    print("Camera intrinsic matrix K:")
    print(k_matrix)

    r_matrix = np.eye(3)
    t_vector = np.array([[0.0], [0.0], [5.0]])

    points_3d = get_cube_points()
    points_2d = project_points(points_3d, k_matrix, r_matrix, t_vector)
    plot_projection(points_2d, points_3d, image_width, image_height)


if __name__ == "__main__":
    # [补全] 增加标准脚本入口，便于直接运行实验。
    main()
