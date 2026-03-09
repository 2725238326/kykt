import matplotlib.pyplot as plt
import numpy as np


def get_cube_points():
    """Create the 8 vertices of a cube centered at the world origin."""
    return np.array(
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


def build_intrinsics_from_phone_metadata(image_width, image_height, focal_length_mm, sensor_width_mm):
    """
    Build an approximate intrinsic matrix from phone metadata.

    Notes:
    - image_width / image_height come from the real photo resolution.
    - focal_length_mm here should be the physical focal length if known.
    - sensor_width_mm is still an approximation unless measured or calibrated.
    """
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def project_points(points_world, k_matrix, r_matrix, t_vector):
    """Project 3D world points into 2D pixel coordinates."""
    points_cam = r_matrix @ points_world + t_vector

    valid_depth = points_cam[2, :] > 0
    if not np.all(valid_depth):
        count = int(np.sum(~valid_depth))
        print(f"Warning: {count} point(s) are on or behind the camera plane.")

    points_img_homo = k_matrix @ points_cam
    u = points_img_homo[0, :] / points_img_homo[2, :]
    v = points_img_homo[1, :] / points_img_homo[2, :]
    return np.vstack((u, v))


def plot_projection(points_2d, points_3d, image_width, image_height):
    """Show the projected 2D points and the 3D scene."""
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

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Approx Real-Camera Projection")
    plt.scatter(points_2d[0, :], points_2d[1, :], c="red", s=40)
    for p1, p2 in edges:
        plt.plot(
            [points_2d[0, p1], points_2d[0, p2]],
            [points_2d[1, p1], points_2d[1, p2]],
            color="dodgerblue",
            linewidth=2,
        )
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)

    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.set_title("World View")
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c="royalblue")
    ax.scatter(0, 0, -5, c="black", marker="^", s=100, label="Camera")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def main():
    # Real metadata from your iPhone 15 Pro main-camera photos:
    # resolution: 5712 x 4284
    # labeled lens: 24 mm
    #
    # Important:
    # "24 mm" in the phone UI is typically 35mm-equivalent focal length,
    # not the true physical focal length used directly in the pinhole model.
    # So below we use an approximate physical focal length and sensor width
    # only for teaching/demo purposes.
    image_width = 5712
    image_height = 4284

    # Approximation for a phone main camera.
    # These are not calibration results.
    physical_focal_length_mm = 6.8
    sensor_width_mm = 9.8

    k_matrix = build_intrinsics_from_phone_metadata(
        image_width=image_width,
        image_height=image_height,
        focal_length_mm=physical_focal_length_mm,
        sensor_width_mm=sensor_width_mm,
    )

    print("Approx intrinsic matrix K (phone-inspired, not calibrated):")
    print(k_matrix)
    print()
    print("Metadata used from your photo:")
    print(f"  image size: {image_width} x {image_height}")
    print("  labeled camera: iPhone 15 Pro main camera (24 mm)")
    print()
    print("Approx assumptions used in code:")
    print(f"  physical focal length ~= {physical_focal_length_mm} mm")
    print(f"  sensor width ~= {sensor_width_mm} mm")

    r_matrix = np.eye(3)
    t_vector = np.array([[0.0], [0.0], [5.0]])

    points_3d = get_cube_points()
    points_2d = project_points(points_3d, k_matrix, r_matrix, t_vector)
    plot_projection(points_2d, points_3d, image_width, image_height)


if __name__ == "__main__":
    main()
