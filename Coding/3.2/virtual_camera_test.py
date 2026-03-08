import matplotlib.pyplot as plt
import numpy as np


def get_cube_points():
    """Create the 8 vertices of a cube centered at the world origin."""
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
    """Build the camera intrinsic matrix K."""
    f_pixels = (f_mm / sensor_w_mm) * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0

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
    """Project 3D world points into 2D pixel coordinates."""
    points_cam = r_matrix @ points_world + t_vector

    negative_depths = points_cam[2, :] <= 0
    if np.any(negative_depths):
        count = int(np.sum(negative_depths))
        print(f"Warning: {count} point(s) are on or behind the camera plane.")

    points_img_homo = k_matrix @ points_cam
    u = points_img_homo[0, :] / points_img_homo[2, :]
    v = points_img_homo[1, :] / points_img_homo[2, :]
    return np.vstack((u, v))


def plot_projection(points_2d, points_3d, image_width, image_height):
    """Show the projected 2D image points and the 3D scene."""
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
    main()
