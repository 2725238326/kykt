import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_cube_points():
    """
    生成一个简单的 3D 立方体顶点坐标，用于测试。
    立方体中心在原点 (0,0,0)，边长为 2。
    """
    points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], # 后底面
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]   # 前顶面
    ]).T # 转置为 (3, N) 格式，方便矩阵乘法
    return points

def get_camera_intrinsics(image_width, image_height, f_mm, sensor_w_mm):
    """
    计算内参矩阵 K
    f_mm: 焦距 (毫米)
    sensor_w_mm: 传感器宽度 (毫米)
    """
    # 1. 计算像素焦距 (f_x, f_y)
    # 假设像素是正方形
    f_pixels = (f_mm / sensor_w_mm) * image_width
    
    # 2. 计算主点 (cx, cy) - 通常在图像中心
    cx = image_width / 2
    cy = image_height / 2
    
    # 3. 构造内参矩阵 K (3x3)
    K = []
    raise NotImplementedError("请构建内存矩阵")

    return K

def project_points(points_world, K, R, t):
    """
    核心函数：将 3D 世界坐标投影到 2D 像素坐标
    
    参数:
    points_world: (3, N) 3D点云
    K: (3, 3) 内参矩阵
    R: (3, 3) 旋转矩阵 (World -> Camera)
    t: (3, 1) 平移向量 (World -> Camera)
    
    返回:
    points_pixel: (2, N) 2D像素坐标
    """
    # -----------------------------------------------------------
    # 步骤 1: 世界坐标系 -> 相机坐标系
    # P_cam = R * P_world + t
    # -----------------------------------------------------------
    # 注意 NumPy 的广播机制，t 会自动加到每一列上
    points_cam = np.dot(R, points_world) + t
    

    # 在真实 SfM 中，这步用于剔除错误的解
    negative_depths = points_cam[2, :] < 0
    if np.any(negative_depths):
        print(f"警告: 有 {np.sum(negative_depths)} 个点位于相机背面！")

    # -----------------------------------------------------------
    # 步骤 2: 相机坐标系 -> 图像物理坐标 (归一化平面) -> 像素坐标
    # P_img_homo = K * P_cam
    # -----------------------------------------------------------
    points_img_homo = np.dot(K, points_cam) # 结果形状 (3, N)
    
    # -----------------------------------------------------------
    # 步骤 3: 透视除法 (归一化)
    # u = x' / z', v = y' / z'
    # -----------------------------------------------------------
    u = v = 0
    raise NotImplementedError("请完成透视除法")
    return np.vstack((u, v))

# ================= 主程序 =================

# 1. 设置图像参数
W, H = 800, 600
K = get_camera_intrinsics(W, H, f_mm=35, sensor_w_mm=32)
print("内参矩阵 K:\n", K)

# 2. 设置外参
# 场景：相机位于 Z轴上的 -5 处，看着原点
# 旋转矩阵 R: 单位矩阵 (没有旋转)
R = np.eye(3) 
# 平移向量 t: 将世界原点搬到相机坐标系下
# 如果相机在 (0, 0, -5)，那么世界原点相对于相机就是在 (0, 0, 5)
t = np.array([[0], [0], [5.0]]) 

# 3. 获取 3D 数据
points_3d = get_cube_points()

# 4. 执行投影
points_2d = project_points(points_3d, K, R, t)

# 5. 可视化
plt.figure(figsize=(10, 5))

# 左图：2D 投影结果
plt.subplot(1, 2, 1)
plt.title("Camera View (2D Image)")
plt.scatter(points_2d[0, :], points_2d[1, :], c='r', marker='o')
# 画出连线方便观察立方体结构
edges = [
    (0,1), (1,2), (2,3), (3,0), # 后底面
    (4,5), (5,6), (6,7), (7,4), # 前顶面
    (0,4), (1,5), (2,6), (3,7)  # 连接前后
]
for p1, p2 in edges:
    plt.plot([points_2d[0, p1], points_2d[0, p2]], 
             [points_2d[1, p1], points_2d[1, p2]], 'b-')

plt.xlim(0, W)
plt.ylim(H, 0) # 图像坐标系原点在左上角，Y轴向下
plt.gca().set_aspect('equal')
plt.grid(True)

# 右图：3D 场景示意 (上帝视角)
ax = plt.subplot(1, 2, 2, projection='3d')
ax.set_title("World View (3D Scene)")
ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c='b')
# 画出相机位置 (简单示意)
ax.scatter(0, 0, -5, c='k', marker='^', s=100, label="Camera")
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()

# Q1 修改焦距：把 f_mm 从 35mm 改成 100mm，图像上的立方体会变大还是变小？为什么？
# Q2 修改相机位置：把 t 改成 [1, 0, 5] (相机向左平移)，图像中的立方体会向哪边移动？（这能帮助他们理解相机移动与图像移动的反向关系）。
# Q3 坐标系陷阱：如果不做 plt.ylim(H, 0) 翻转 Y 轴，图像看起来是正的还是倒的？这说明了图像坐标系和数学笛卡尔坐标系的什么区别？
# Q4 目前的相机是正对 Z 轴的。试着构造一个 R，让相机绕 Y 轴旋转 45 度。查找 "Rotation Matrix Y-axis formula"，然后用 np.array 实现它