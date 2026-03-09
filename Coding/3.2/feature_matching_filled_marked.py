from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images(path1, path2):
    """读取两张图像并转换为灰度图。"""
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    # [补全] 增加文件读取失败检查，避免路径错误时程序直接崩溃。
    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"无法读取图片：\n  {path1}\n  {path2}\n"
            "请确认图片存在，或修改 main() 中的路径。"
        )

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2


def detect_and_describe(gray_img):
    """使用 SIFT 提取关键点和描述子。"""
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img, None)
    return kp, des


def match_features(des1, des2, ratio=0.75):
    """使用 KNN 匹配并通过 Lowe Ratio Test 进行初筛。"""
    # [补全] 增加空描述子保护。
    if des1 is None or des2 is None:
        return []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        # [补全] 防止个别匹配对长度不足 2。
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def geometric_verification(kp1, kp2, good_matches):
    """使用 RANSAC 估计基础矩阵并返回内点掩码。"""
    if len(good_matches) < 8:
        print("匹配点不足，无法估计基础矩阵。")
        return None, None, good_matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # [补全] 使用 OpenCV 的 RANSAC 方法估计基础矩阵 F。
    f_matrix, mask = cv2.findFundamentalMat(
        src_pts,
        dst_pts,
        cv2.FM_RANSAC,
        3.0,
        0.99,
    )

    # [补全] 增加 RANSAC 失败保护。
    if f_matrix is None or mask is None:
        print("RANSAC 未能估计出有效的基础矩阵。")
        return None, None, good_matches

    matches_mask = mask.ravel().tolist()
    return f_matrix, matches_mask, good_matches


def visualize_matches(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    mask=None,
    title="Matches",
    max_matches=50,
    save_path=None,
):
    """可视化匹配结果。"""
    # [补全] 如果传入的是 RANSAC 掩码，这里先只保留内点。
    if mask is not None:
        filtered_pairs = [(m, keep) for m, keep in zip(matches, mask) if keep]
        matches = [m for m, _ in filtered_pairs]
        mask = None

    # [补全] 按匹配距离排序，优先显示质量更好的匹配。
    matches = sorted(matches, key=lambda m: m.distance)

    # [补全] 只显示前 max_matches 个匹配，便于观察结果。
    if max_matches is not None:
        matches = matches[:max_matches]

    draw_params = dict(
        # [补全] 将匹配线改成更醒目的蓝色，并加粗线条，方便展示。
        matchColor=(255, 0, 0),
        singlePointColor=(255, 215, 0),
        matchesMask=mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
        matchesThickness=2,
    )

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # [补全] 保存实验结果图，方便写报告和展示。
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent

    # [补全] 这里把默认图片改成了当前实验效果最好的一组。
    img1_path = script_dir / "2-a.jpg"
    img2_path = script_dir / "2-b.jpg"

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
        max_matches=80,
        save_path=script_dir / "matches_before_ransac.png",
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
        max_matches=80,
        save_path=script_dir / "matches_after_ransac.png",
    )


if __name__ == "__main__":
    # [补全] 增加标准脚本入口，便于独立运行。
    main()
