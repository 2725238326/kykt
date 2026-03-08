import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(path1, path2):
    """读取图像并转换为灰度图"""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("无法找到图像文件，请检查路径！")

    # SIFT 等特征提取通常在灰度图上进行
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2

def detect_and_describe(gray_img):
    """
    使用 SIFT 提取特征点 (Keypoints) 和描述子 (Descriptors)
    """
    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create()
    
    # 检测并计算
    # kp: 关键点列表 (包含位置、尺度、方向)
    # des: 描述子矩阵 (N x 128)
    kp, des = sift.detectAndCompute(gray_img, None)
    return kp, des

def match_features(des1, des2):
    """
    使用 KNN (K-Nearest Neighbors) 进行特征匹配
    并使用 Lowe's Ratio Test 初步过滤
    """
    # 暴力匹配器 (Brute-Force Matcher)
    bf = cv2.BFMatcher()
    
    # 对每个 query (des1中的点)，在 train (des2) 中找 2 个最近邻
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    # Lowe's Ratio Test:
    # 如果最近邻距离 < 0.75 * 次近邻距离，则认为匹配可靠
    # 这是一个经验法则：好的匹配应该是独一无二的
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    return good_matches

def geometric_verification(kp1, kp2, good_matches):
    """
    关键步骤：利用 RANSAC 计算基础矩阵 (Fundamental Matrix)
    并剔除不符合几何约束的误匹配 (Outliers)
    """
    if len(good_matches) < 8:
        print("匹配点太少，无法计算基础矩阵！")
        return None, None, None

    # 1. 提取匹配点的坐标 (x, y)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 2. 计算基础矩阵 F，使用 RANSAC 算法
    # cv2.FM_RANSAC: 使用 RANSAC 算法
    # 3.0: 距离阈值 (像素单位)。点到极线的距离超过 3 像素就被认为是 Outlier
    # 0.99: 置信度
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)

    # mask 包含了哪些点是 Inlier (1)，哪些是 Outlier (0)
    matches_mask = mask.ravel().tolist()
    
    return F, matches_mask, good_matches

def visualize_matches(img1, kp1, img2, kp2, matches, mask=None, title="Matches"):
    """可视化匹配结果"""
    # 绘图参数
    draw_params = dict(matchColor=(0, 255, 0), # 绿色表示匹配线
                       singlePointColor=None,
                       matchesMask=mask, # 只画 Inliers
                       flags=2)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ================= 主程序 =================

# 1. 加载图像 (替换为你自己的图片路径)
# 如果没有图片，建议让学生用手机现拍两张
img1, img2, gray1, gray2 = load_images('leuvenA.jpg', 'leuvenB.jpg')

# 2. 提取特征
print("正在提取特征...")
kp1, des1 = detect_and_describe(gray1)
kp2, des2 = detect_and_describe(gray2)
print(f"图1 特征点数: {len(kp1)}, 图2 特征点数: {len(kp2)}")

# 3. 初始匹配 (Ratio Test)
print("正在进行匹配...")
good_matches = match_features(des1, des2)
print(f"Ratio Test 后保留匹配数: {len(good_matches)}")

# 可视化阶段 1: 仅做 Ratio Test 的结果 (通常包含错误的交叉连线)
visualize_matches(img1, kp1, img2, kp2, good_matches, title="After Ratio Test (No RANSAC)", N=10)

# 4. 几何验证 (RANSAC)
print("正在运行 RANSAC...")
F, matches_mask, final_matches = geometric_verification(kp1, kp2, good_matches)

# 统计 RANSAC 结果
inlier_count = matches_mask.count(1)
print(f"RANSAC 后最终 Inliers 数: {inlier_count}")
print(f"剔除率: {(1 - inlier_count/len(good_matches))*100:.2f}%")
print("基础矩阵 F:\n", F)

# 5. 可视化最终结果
visualize_matches(img1, kp1, img2, kp2, final_matches, mask=matches_mask, title="Final Matches (After RANSAC)", N=10)