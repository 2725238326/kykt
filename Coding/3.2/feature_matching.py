from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images(path1, path2):
    """Read two images and convert them to grayscale."""
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Unable to read images:\n  {path1}\n  {path2}\n"
            "Please place two sample images in the same folder as this script "
            "or update the paths in main()."
        )

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2


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


def geometric_verification(kp1, kp2, good_matches):
    """Estimate the fundamental matrix with RANSAC and return the inlier mask."""
    if len(good_matches) < 8:
        print("Not enough matches to estimate the fundamental matrix.")
        return None, None, good_matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    f_matrix, mask = cv2.findFundamentalMat(
        src_pts,
        dst_pts,
        cv2.FM_RANSAC,
        3.0,
        0.99,
    )

    if f_matrix is None or mask is None:
        print("RANSAC failed to estimate a valid fundamental matrix.")
        return None, None, good_matches

    matches_mask = mask.ravel().tolist()
    return f_matrix, matches_mask, good_matches


def visualize_matches(img1, kp1, img2, kp2, matches, mask=None, title="Matches", max_matches=50):
    """Visualize a subset of matches."""
    if max_matches is not None:
        matches = matches[:max_matches]
        if mask is not None:
            mask = mask[:max_matches]

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent
    img1_path = script_dir / "leuvenA.jpg"
    img2_path = script_dir / "leuvenB.jpg"

    print("Loading images...")
    img1, img2, gray1, gray2 = load_images(img1_path, img2_path)

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
        title="After Ratio Test (Before RANSAC)",
        max_matches=50,
    )

    print("Running RANSAC...")
    f_matrix, matches_mask, final_matches = geometric_verification(kp1, kp2, good_matches)

    if matches_mask is None:
        return

    inlier_count = matches_mask.count(1)
    outlier_ratio = 1 - inlier_count / len(good_matches)
    print(f"Inliers after RANSAC: {inlier_count}")
    print(f"Rejected ratio: {outlier_ratio * 100:.2f}%")
    print("Fundamental matrix F:")
    print(f_matrix)

    visualize_matches(
        img1,
        kp1,
        img2,
        kp2,
        final_matches,
        mask=matches_mask,
        title="Final Matches (After RANSAC)",
        max_matches=50,
    )


if __name__ == "__main__":
    main()
