import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 
from scipy.optimize import least_squares
from tqdm import tqdm 

class ImageLoader:
    """
    【数据加载模块】
    负责从指定目录加载图片，并根据缩放比例 (downscale_factor) 同步缩放图像尺寸和相机内参 (K矩阵)。
    图像缩小一倍，内参矩阵中的焦距 (fx, fy) 和光心坐标 (cx, cy) 也必须等比例缩小。
    """
    def __init__(self, img_dir:str, downscale_factor:float):
        # 确保目录路径为绝对路径
        self.img_dir= os.path.abspath(img_dir)

        # 加载相机内参矩阵 (K.txt)
        # 输入：包含 9 个浮点数的 3x3 矩阵文本
        # 输出：self.K (3x3 np.ndarray)
        k_file = os.path.join(self.img_dir, 'K.txt')
        with open(k_file, 'r') as f:
            lines= f.read().strip().split('\n')
            matrix_values=[]
            for line in lines:
                row_vals= [float(val) for val in line.strip().split()]
                matrix_values.append(row_vals)
            self.K =np.array(matrix_values, dtype=np.float32) #3x3 

        # 收集图片文件路径
        self.image_list=[]
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg','.png')):
                self.image_list.append(os.path.join(self.img_dir,filename))

        self.path = os.getcwd()
        self.factor = downscale_factor

        # 调整内参矩阵以适应缩放后的图像
        self.downscale_instrinsics()

    def downscale_image(self, image):
        """ 按比例缩小图像，减少后续计算量 """
        new_w= int(image.shape[1]/ self.factor)
        new_h= int(image.shape[0]/ self.factor)
        return cv2.resize(image,(new_w,new_h), interpolation=cv2.INTER_LINEAR)

    def downscale_instrinsics(self) -> None:
        """ 同步缩放相机内参的焦距和主点坐标 """
        self.K[0, 0] /= self.factor #fx (X轴焦距)
        self.K[1, 1] /= self.factor #fy (Y轴焦距)
        self.K[0, 2] /= self.factor #cx (光心X坐标)
        self.K[1, 2] /= self.factor #cy (光心Y坐标)

    
class StructurefromMotion:
    """
    【核心算法模块：运动恢复结构 (SfM)】
    流程概览：
    1. 提取并匹配特征点。
    2. 使用前两张图计算本质矩阵，恢复初始相机位姿（双视图初始化）。
    3. 三角化算出第一批 3D 点云。
    4. 逐张加入新图片，利用已知 3D 点求解新相机位姿 (PnP)。
    5. 使用新相机位姿，三角化出更多新的 3D 点。
    6. 执行光束法平差 (Bundle Adjustment) 消除累积误差。
    """

    def __init__(self, img_dir=str, downscale_factor:float = 2.0):
        self.img_obj =ImageLoader(img_dir, downscale_factor)

    def feature_matching(self, image_0, image_1) -> tuple:
        """
        【特征提取与匹配】
        量化局部特征，并计算描述符之间的数学距离以建立 2D 像素的对应关系。
        
        输入：两张 BGR 图像
        输出：两组 Nx2 的 numpy 数组，代表在两张图中成功匹配的 2D 像素坐标 (pts0, pts1)
        """
        # 1. 提取 SIFT 特征：将像素强度模式量化为描述符向量
        sift = cv2.SIFT_create(nfeatures=10000) 
        key_points0, descriptors_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points1, descriptors_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        # 2. FLANN 快速最近邻搜索比对
        index_params= dict(algorithm=1, trees=15)
        search_params=dict(checks=200)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors_0, descriptors_1, k=2)

        # 3. Lowe's ratio test：通过距离比值剔除模糊匹配
        ratio_thresh=0.70
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # 提取通过筛选的特征点坐标
        pts0=np.float32([key_points0[m.queryIdx].pt for m in good_matches]) 
        pts1=np.float32([key_points1[m.trainIdx].pt for m in good_matches])

        return pts0, pts1
    
    def triangulation(self,proj_matrix_1, proj_matrix_2, pts_2d_1, pts_2d_2) -> tuple:
        """
        【三角化测量 (Triangulation)】
        已知两个相机的空间位置，以及同一个物体在两张照片上的 2D 像素坐标，求光线交点，算出真实 3D 坐标。
                
        输入：
            proj_matrix_1, proj_matrix_2: 两个相机的 3x4 投影矩阵 (P = K[R|t])
            pts_2d_1, pts_2d_2: 匹配好的二维点集 (Nx2)
        输出：
            计算出的三维空间点云 (Nx3 的笛卡尔坐标)
        """
        # 输入形状需转置为 2xN，输出为 4xN 的齐次坐标 (X, Y, Z, W)
        point_cloud = cv2.triangulatePoints(proj_matrix_1, proj_matrix_2, pts_2d_1.T, pts_2d_2.T)
        
        # 齐次坐标转为 3D 笛卡尔坐标：除以最后一位的尺度因子 W
        return pts_2d_1.T, pts_2d_2.T, (point_cloud/point_cloud[3])
    
    def solve_PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        【求解新相机位姿 (PnP 问题)】
        拿着已知的 3D 坐标去反推新加入相机的拍摄位置（旋转 R 和平移 t）。
                
        输入：
            obj_point: 已知的 3D 空间点坐标 (Nx3)
            image_point: 这些 3D 点在新图像上对应的 2D 像素坐标 (Nx2)
            K: 相机内参
        输出：
            rot_matrix: 3x3 旋转矩阵 (表示相机朝向)
            tran_vector: 3x1 平移向量 (表示相机位置)
            inlier: 经过 RANSAC 过滤后真正正确的匹配点集
        """
        if initial == 1:
            obj_point=obj_point[:,0,:]
            image_point = image_point.T
            rot_vector = rot_vector.T

        # 使用带有 RANSAC 的 PnP 算法，不仅解算位姿，还能剔除错误的 3D-2D 对应关系
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        
        # 将 3x1 的旋转向量 (罗德里格斯表示法) 转换为 3x3 的旋转矩阵
        rot_matrix, _ =cv2.Rodrigues(rot_vector_calc)

        # 仅保留被判定为正确的内点 (Inliers)
        if inlier is not None:
            image_point=image_point[inlier[:,0]]
            obj_point=obj_point[inlier[:,0]]
            rot_vector = rot_vector[inlier[:,0]]
        return rot_matrix, tran_vector,image_point,obj_point,rot_vector
    
    def find_common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        【数据关联 (Data Association)：寻找 2D-3D 对应关系与新特征点】
        
        核心目的：
        在增量式 SfM (如处理连续的三张图 A -> B -> C) 中，图 B 起到了承前启后的作用。
        我们需要在图 B 刚刚与图 C 建立的新匹配点中，鉴别出哪些点是：
        1. 「老住户 (旧点)」：它们既在 A-B 中匹配成功(已通过三角化生成了 3D 坐标)，又在 B-C 中匹配成功。
           - 作用：这些点拥有已知的 3D 坐标 和 图 C(新图) 上的 2D 成像坐标，正好凑齐了 PnP 算法的求解条件，可以用来反算图 C 的摄像机姿态。
        2. 「新住户 (新点)」：它们仅在 B-C 中首次匹配成功，在 A-B 的匹配中未出现(尚无 3D 坐标)。
           - 作用：等我们用前面的“老住户”通过 PnP 算出图 C 的相机位姿后，再利用已知的 图 B 和 图 C 相机姿态，对这些“新住户”进行双视图三角化，生成全新的 3D 坐标并加入点云，供下一张图(图 D)定位使用。

        输入参数：
        - image_points_1 (图 B 的特征点集, Nx2): 它们来自上一轮(图 A-B)匹配，且每一个点都已经通过三角化拥有了对应的 3D 空间坐标。
        - image_points_2 (图 B 的特征点集, Mx2): 它们来自当前轮(图 B-C)刚刚完成的特征匹配。
        - image_points_3 (图 C 的特征点集, Mx2): 这也是来自当前轮(图 B-C)的匹配，这些点分别与 image_points_2 中的点在 2D 像素位置上一一对应。

        输出：
        - cm_points_1 (索引数组): 「老住户」在 image_points_1 (包含 3D 坐标的旧集合) 中的索引位置。
        - cm_points_2 (索引数组): 「老住户」在 image_points_2 与 image_points_3 (当前新匹配的集合) 中的索引位置。
        - mask_array_1 (图 B 的 2D 坐标数组): 过滤掉老住户后，剩下的**纯纯的「新住户」**在图 B 上的 2D 坐标。(供后续三角化使用)
        - mask_array_2 (图 C 的 2D 坐标数组): 同上，与 mask_array_1 一一对应的「新住户」在图 C 上的 2D 坐标。
        """
        cm_points_1 = []
        cm_points_2 = []
        
        # 1. 遍历含有 3D 坐标的“旧特征点集”(image_points_1)
        # 用 np.where 逐一去“新建立的匹配集”(image_points_2) 中搜索是否重合。
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                # 如果这个坐标既出现在旧匹配堆里，也出现在新匹配堆里，那就是我们要找的“老住户(交集点)”
                cm_points_1.append(i)         # 记录在旧集合(自带3D坐标)中的位置
                cm_points_2.append(a[0][0])   # 记录在新匹配对(等待PnP)中的位置

        # 2. 剥离与过滤：把新匹配堆里找到的那些“老住户”屏蔽掉，剩下的就是亟待发往【三角化部门】重建 3D 坐标的“纯新点”
        # 使用 numpy 的掩码数组 (Masked Array) 技术进行高效剔除。
        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True # 将老住户标记为需要屏蔽 (True)
        mask_array_1 = mask_array_1.compressed() # 压缩数组：直接把标记为 True 的点丢掉
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0]/2), 2) # 恢复 Nx2 的二维坐标格式

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True # 对图 C 侧对应的点位进行同样的屏蔽操作
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        
        print(" Shape of New Array", mask_array_1.shape, mask_array_2.shape)
        
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def reproj_error(self, obj_points, image_points, transform_matrix, K, homogenity) -> tuple:
        """
        【计算重投影误差】
        将算出来的 3D 点重新投射回相机屏幕，与实际提取的 2D 特征点计算距离差。
        误差越小，说明位姿和 3D 点算得越准。
        """
        rot_matrix = transform_matrix[:3,:3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        if homogenity == 1:
            obj_points= cv2.convertPointsFromHomogeneous(obj_points.T)

        # 核心算式：利用算出的 R, t, K 将 3D 点拍扁成理论上的 2D 像素点
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:,0,:])

        # 计算理论投影位置与真实观测位置的 L2 距离 (欧氏距离)
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error/ len(image_points_calc), obj_points
            
    def optimize_reproj_error(self, obj_points) -> np.array:
        """ 【光束法平差(BA) 目标函数】：定义优化器需要最小化的误差公式 """
        transform_matrix = obj_points[0:12].reshape((3,4)) 
        K = obj_points[12:21].reshape((3,3)) 
        rest= int(len(obj_points[21:])* 0.4) 
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3),3)) 

        rot_matrix = transform_matrix[:3,:3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector , _ = cv2.Rodrigues(rot_matrix)

        image_points ,_ =cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:,0,:]
        
        # 返回误差数组：真实坐标 p 减去 理论投影 image_points
        error = [(p[idx]- image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)
    
    def compute_bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        """
        【光束法平差 (Bundle Adjustment)】
        非线性优化过程：同时微调相机位姿 (R,t) 和 3D 坐标，使得全局重投影误差最小。
        """
        # 将所有需要优化的参数打包成一维长数组 (Scipy 优化器的标准格式)
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables= np.hstack((opt_variables, opt.ravel()))
        opt_variables= np.hstack((opt_variables, _3d_point.ravel()))

        # 调用 scipy 的 least_squares 最小二乘求解器
        values_corrected = least_squares(self.optimize_reproj_error, opt_variables, gtol= r_error).x
        
        # 优化完毕后，将数据解包还原
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:])* 0.4)
        return (values_corrected[21+rest:].reshape((int(len(values_corrected[21+rest:])/3),3)), 
                values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, 
                values_corrected[0:12].reshape((3,4)))

    def save_to_ply(self, path, point_cloud, colors=None, bundle_adjustment_enabled=False,
                     binary_format=False, scaling_factor=1.0):
        """ 【数据导出】保存最终的 3D 点云与颜色到标准 .ply 格式文件 """
        sub_dir = 'Results with Bundle Adjustment' if bundle_adjustment_enabled else 'Results'
        output_dir = os.path.join(path, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        dataset_name = os.path.basename(os.path.normpath(self.img_obj.img_dir))
        ply_filename = os.path.join(output_dir, f"{dataset_name}.ply")

        point_cloud= np.asarray(point_cloud).reshape(-1,3) * scaling_factor

        if colors is not None:
            colors=np.asarray(colors).reshape(-1,3)
            colors=np.clip(colors,0,255).astype(np.uint8) 
        else:
            colors = np.full_like(point_cloud, fill_value=105, dype=np.uint8) 

        # 对点云进行去均值中心化和尺度归一化
        mean= np.mean(point_cloud,axis=0)
        point_cloud-= mean 
        scale_factor= np.max(np.linalg.norm(point_cloud,axis=1))
        point_cloud/=scale_factor 

        # 统计学过滤：使用 Z-score 剔除偏离主体太远的噪点
        distances =np.linalg.norm(point_cloud,axis=1)
        z_scores= (distances - np.mean(distances))/ np.std(distances)
        mask =np.abs(z_scores) < 2.5 
        point_cloud =point_cloud[mask]
        point_cloud =point_cloud * scale_factor
        colors = colors[mask]

        vertices= np.hstack([point_cloud,colors])

        with open(ply_filename, 'wb' if binary_format else 'w') as f:
            f.write(b'ply\n' if binary_format else 'ply\n')
            f.write(b'format binary_little_endian 1.0\n' if binary_format else 'format ascii 1.0\n')
            f.write(f'element vertex {len(vertices)}\n'.encode())
            f.write(b'property float x\nproperty float y\nproperty float z\n')
            f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(b'end_header\n')

            if binary_format:
                vertices_binary = np.zeros((len(vertices),), dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                ])
                vertices_binary['x'] = vertices[:, 0]
                vertices_binary['y'] = vertices[:, 1]
                vertices_binary['z'] = vertices[:, 2]
                vertices_binary['red'] = vertices[:, 3]
                vertices_binary['green'] = vertices[:, 4]
                vertices_binary['blue'] = vertices[:, 5]
                vertices_binary.tofile(f)
            else:
                np.savetxt(f, vertices, fmt='%f %f %f %d %d %d')

            print(f'Point cloud saved to {ply_filename}')

    def __call__(self, bundle_adjustment_enabled: bool = False):
        """
        【SfM 主循环】
        """
        # 初始化相机0 (世界原点)，其旋转为单位阵，平移为0向量
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0]])
        transform_matrix_1 = np.empty((3,4))
        print('Camera Intrinsic Matrix:', self.img_obj.K)

        # 投影矩阵 P0 = K[I|0]
        pose_0= np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3,4))
        total_points= np.zeros((1,3))
        total_colors = np.zeros((1,3))

        # ==================================
        # 阶段一：双视图初始化 (处理第1张和第2张图)
        # ==================================
        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        features_0, features_1 = self.feature_matching(image_0, image_1)

        # 构建代数方程：求解本质矩阵 (Essential Matrix)
        # 输入：匹配好的二维像素点，内参K
        # 输出：本质矩阵E，验证匹配是否正确的掩码 em_mask
        essential_matrix, em_mask = cv2.findEssentialMat(features_0, features_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        features_0 = features_0[em_mask.ravel()==1]
        features_1 = features_1[em_mask.ravel()==1]

        # 矩阵分解 (SVD)：从本质矩阵 E 中提取相机1的相对位姿 (旋转 R 和平移 t)
        _, rot_matrix, tran_matrix , em_mask = cv2.recoverPose(essential_matrix, features_0, features_1, self.img_obj.K)
        features_0 = features_0[em_mask.ravel()>0]
        features_1 = features_1[em_mask.ravel() > 0] 

        # 构建相机1的 3x4 外参矩阵 [R|t]
        transform_matrix_1[:3, :3]= np.matmul(rot_matrix, transform_matrix_0[:3,:3])
        transform_matrix_1[:3,3]= transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3,:3], tran_matrix.ravel())

        # 相机1的完整投影矩阵 P1 = K[R|t]
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        # 执行视线交汇：三角化算出初始的 3D 空间点云
        features_0, features_1, points_3d = self.triangulation(pose_0, pose_1, features_0, features_1)

        error, points_3d= self.reproj_error(points_3d, features_1, transform_matrix_1, self.img_obj.K, homogenity=1)
        print("Reprojection error for first two images:", error)

        # 使用第一批 3D 点进行一次 PnP 确认坐标系
        _,_, features_1, points_3d, _ = self.solve_PnP(points_3d, features_1, self.img_obj.K,
                                                       np.zeros((5,1), dtype=np.float32), features_0, initial=1)

        total_images = len(self.img_obj.image_list) -2
        print('total_images', total_images)

        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))
        threshold = 0.75

        # ==================================
        # 阶段二：增量式重建 (逐张加入后续照片)
        # ==================================
        for i in tqdm(range(total_images)):
            # 加载新照片
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i+2]))

            # 计算新照片(当前)与上一张照片的特征匹配
            features_cur, features_2 = self.feature_matching(image_1, image_2)
            
            if i !=0:
                features_0 , features_1, points_3d = self. triangulation(pose_0, pose_1, features_0, features_1)
                features_1= features_1.T
                points_3d= cv2.convertPointsFromHomogeneous(points_3d.T)
                
            # 分离出：旧的 3D 点对应的数据用于 PnP，没建过 3D 点的新数据用于后续三角化
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1= self.find_common_points(features_1,features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]
            
            # 使用已有的 3D 点(points_3d)反算当前这张新图片的拍摄姿态 (rot_matrix, tran_matrix)
            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.solve_PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            
            # 记录新相机位姿
            transform_matrix_1= np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            error, points_3d= self.reproj_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity=0)
            
            # 相机位姿已经算准了，现在对尚未有 3D 坐标的新特征点(cm_mask)进行三角化补充
            cm_mask_0, cm_mask_1, points_3d= self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=1)
            print("Reprojection error:", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            # ==================================
            # 阶段三：光束法平差优化 (可选)
            # ==================================
            if bundle_adjustment_enabled:
                # 传入微调所有参数，使得误差 error 下降
                points_3d, cm_mask_1, transform_matrix_1 = self.compute_bundle_adjustment(points_3d, cm_mask_1,
                                                                                          transform_matrix_1, self.img_obj.K,
                                                                                          threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reproj_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Reprojection error after Bundle Adjustment: ",error)

                # 将本轮优化的 3D 点和对应的 2D 像素颜色存入总账
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                # 拾取像素颜色 (用于彩色 PLY 显示)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector)) 

            # 迭代更替：将当前的位姿变为“历史”，开始处理下一张
            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            features_0 = np.copy(features_cur)
            features_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)

            # 实时的特征点提取进度展示
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        # 保存误差下降曲线图
        if bundle_adjustment_enabled:
            plot_dir = os.path.join(self.img_obj.path, 'Results with Bundle Adjustment')
        else:
            plot_dir = os.path.join(self.img_obj.path, 'Results')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Image Index')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error Plot')
        plt.savefig(os.path.join(plot_dir, 'reprojection_errors.png'))
        plt.close()

        if total_points.size == 0 or total_colors.size == 0:
            print("Error: No points or colors to save. Skipping point cloud generation.")
        else:
            print(f"Total points to save: {total_points.shape[0]}")
            print(f"Total colors to save: {total_colors.shape[0]}")
        
        # 导出彩色点云文件
        scaling_factor=5000.0
        self.save_to_ply(self.img_obj.path, total_points, total_colors,
                         bundle_adjustment_enabled, binary_format=True, scaling_factor=scaling_factor)
        print("Saved the point cloud to .ply file!!!")

        results_dir = os.path.join(self.img_obj.path, 'Results Array')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        parent_folder= os.path.basename(os.path.dirname(self.img_obj.image_list[0]))
        pose_csv_name= f"{parent_folder}_pose_array.csv"
        pose_csv_path= os.path.join(results_dir, pose_csv_name)

        np.savetxt(pose_csv_path,pose_array, delimiter='\n')

if __name__ == '__main__':
    # 填入包含图像序列与 K.txt 相机内参文件的路径
    sfm = StructurefromMotion("Dataset/Herz-Jesus-P8") 
    # Dataset/fountain-P11
    # Dataset/Herz-Jesus-P8
    # 启动重建流程，bundle_adjustment_enabled=True 将开启优化消除漂移误差
    sfm(bundle_adjustment_enabled=False)