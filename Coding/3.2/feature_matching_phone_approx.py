from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def build_intrinsics_from_phone_metadata(image_width, image_height, focal_length_mm, sensor_width_mm):
    """
    Build an approximate intrinsic matrix from phone metadata.

    Important:
    - This is an approximation for learning/demo.
    - It is not a calibration result.
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


def load_images(path1, path2):
    """Read two images from disk."""
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))
    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Unable to read images:\n  {path1}\n  {path2}\n"
            "Please confirm that both files exist."
        )
    return img1, img2


def undistort_if_needed(img, k_matrix, dist_coeffs=None):
    """
    Undistort an image if distortion coefficients are provided.

    For this demo, dist_coeffs defaults to zeros because we do not have
    a real calibration result yet.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=float)
    return cv2.undistort(img, k_matrix, dist_coeffs)


def detect_and_describe(gray_img):
    """Extract SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors


def match_features(des1, des2, ratio=0.75):
    """Run KNN matching and keep matches that pass Lowe's ratio test."""
    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def estimate_fundamental_matrix(kp1, kp2, matches):
    """Estimate the fundamental matrix using RANSAC."""
    if len(matches) < 8:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    f_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)
    return f_matrix, mask


def estimate_pose_with_intrinsics(kp1, kp2, matches, k_matrix):
    """
    Estimate the essential matrix and relative pose using approximate intrinsics.

    This is the key difference from the plain feature_matching.py version:
    once K is available, we can move from F to E and recover (R, t).
    """
    if len(matches) < 8:
        return None, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    essential_matrix, mask = cv2.findEssentialMat(
        src_pts,
        dst_pts,
        cameraMatrix=k_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if essential_matrix is None or mask is None:
        return None, None, None, None

    inlier_src = src_pts[mask.ravel() == 1]
    inlier_dst = dst_pts[mask.ravel() == 1]
    if len(inlier_src) < 5:
        return essential_matrix, mask, None, None

    _, r_matrix, t_vector, pose_mask = cv2.recoverPose(
        essential_matrix,
        inlier_src,
        inlier_dst,
        k_matrix,
    )
    return essential_matrix, mask, r_matrix, t_vector


def visualize_matches(img1, kp1, img2, kp2, matches, title, mask=None, max_matches=80, save_path=None):
    """Visualize a subset of matches."""
    if mask is not None:
        filtered_pairs = [(m, keep) for m, keep in zip(matches, mask) if keep]
        matches = [m for m, _ in filtered_pairs]

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:max_matches]

    draw_params = dict(
        matchColor=(255, 0, 0),
        singlePointColor=(255, 215, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT,
        matchesThickness=2,
    )

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent

    # Choose two images taken by the same phone camera if possible.
    img1_path = script_dir / "2-a.jpg"
    img2_path = script_dir / "2-b.jpg"

    img1_raw, img2_raw = load_images(img1_path, img2_path)
    image_height, image_width = img1_raw.shape[:2]

    # Phone-inspired approximate intrinsics for iPhone 15 Pro main camera.
    # These are approximate learning values, not calibration results.
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
    print("Using approximate phone-camera assumptions:")
    print(f"  image size: {image_width} x {image_height}")
    print(f"  physical focal length ~= {physical_focal_length_mm} mm")
    print(f"  sensor width ~= {sensor_width_mm} mm")

    img1 = undistort_if_needed(img1_raw, k_matrix)
    img2 = undistort_if_needed(img2_raw, k_matrix)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print()
    print("Detecting SIFT features...")
    kp1, des1 = detect_and_describe(gray1)
    kp2, des2 = detect_and_describe(gray2)
    print(f"Image 1 keypoints: {len(kp1)}")
    print(f"Image 2 keypoints: {len(kp2)}")

    print("Matching descriptors...")
    good_matches = match_features(des1, des2)
    print(f"Matches after ratio test: {len(good_matches)}")

    visualize_matches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        title="Approx Phone Camera: Matches Before Geometry",
        max_matches=80,
        save_path=script_dir / "matches_phone_before_geometry.png",
    )

    print("Estimating F with RANSAC...")
    f_matrix, f_mask = estimate_fundamental_matrix(kp1, kp2, good_matches)
    if f_matrix is not None and f_mask is not None:
        f_inliers = int(f_mask.ravel().sum())
        print(f"Fundamental-matrix inliers: {f_inliers}")
        print("Fundamental matrix F:")
        print(f_matrix)
    else:
        print("Failed to estimate a valid fundamental matrix.")

    print()
    print("Estimating E and relative pose with approximate K...")
    e_matrix, e_mask, r_matrix, t_vector = estimate_pose_with_intrinsics(kp1, kp2, good_matches, k_matrix)
    if e_matrix is None or e_mask is None:
        print("Failed to estimate a valid essential matrix.")
        return

    e_inliers = int(e_mask.ravel().sum())
    print(f"Essential-matrix inliers: {e_inliers}")
    print("Essential matrix E:")
    print(e_matrix)

    visualize_matches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        title="Approx Phone Camera: Inliers After Essential Matrix",
        mask=e_mask.ravel().tolist(),
        max_matches=80,
        save_path=script_dir / "matches_phone_after_essential.png",
    )

    if r_matrix is not None and t_vector is not None:
        print()
        print("Recovered relative rotation R:")
        print(r_matrix)
        print("Recovered relative translation direction t:")
        print(t_vector)
    else:
        print("Pose recovery was not stable enough to return R and t.")


if __name__ == "__main__":
    main()
