import open3d as o3d
import numpy as np
import cv2
import os
from math import pi
import threading


class PointCloudGenerator:
    """
    彩色图像与深度图像转点云的工具类
    使用相机相对于TCP的6D位姿（[x, y, z, rx, ry, rz]）替代变换矩阵
    """

    def __init__(
            self,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            tcp_pose: list = None,
            camera_to_tcp_pose: list = None,  # 相机相对于TCP的6D位姿
            visualize: bool = False,
            save_point_cloud: bool = False,
            save_path: str = "point_cloud.npy"
    ):
        """
        类初始化：配置相机内参、机械臂参数、可视化与保存选项

        Args:
            fx, fy, cx, cy: 相机内参
            tcp_pose: 机械臂TCP在基坐标系下的6D位姿 [x, y, z, rx, ry, rz]
            camera_to_tcp_pose: 相机相对于TCP的6D位姿 [x, y, z, rx, ry, rz]
                               单位：平移(mm)，旋转(弧度)
        """
        # 相机内参（核心参数）
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # 机械臂TCP位姿（默认值）
        self.tcp_pose = tcp_pose if tcp_pose is not None else [
            297.409, 336.295, 651.083, -179.997 / 180 * pi, 0 / 180 * pi, -45.002 / 180 * pi
        ]
        if len(self.tcp_pose) != 6:
            raise ValueError("tcp_pose必须是长度为6的列表：[x, y, z, rx, ry, rz]")

        # 相机相对于TCP的6D位姿（默认值：相机在TCP前方100mm处，无旋转）
        self.camera_to_tcp_pose = camera_to_tcp_pose if camera_to_tcp_pose is not None else [
            0, 0, 100, 0, 0, 0  # x, y, z, rx, ry, rz
        ]
        if len(self.camera_to_tcp_pose) != 6:
            raise ValueError("camera_to_tcp_pose必须是长度为6的列表：[x, y, z, rx, ry, rz]")

        # 可视化与保存配置
        self.visualize = visualize
        self.save_point_cloud = save_point_cloud
        self.save_path = save_path

        # 可视化窗口相关
        self.vis = None
        self.visualization_thread = None

        # 缓存生成的点云数据
        self.generated_point_cloud = None

    @staticmethod
    def rotation_matrix(r_x: float, r_y: float, r_z: float) -> np.ndarray:
        """生成绕X、Y、Z轴旋转的组合旋转矩阵（Z→Y→X顺序）"""
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r_x), -np.sin(r_x)],
                       [0, np.sin(r_x), np.cos(r_x)]])
        Ry = np.array([[np.cos(r_y), 0, np.sin(r_y)],
                       [0, 1, 0],
                       [-np.sin(r_y), 0, np.cos(r_y)]])
        Rz = np.array([[np.cos(r_z), -np.sin(r_z), 0],
                       [np.sin(r_z), np.cos(r_z), 0],
                       [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    @staticmethod
    def pose_to_matrix(pose: list) -> np.ndarray:
        """将6D位姿 [x,y,z,rx,ry,rz] 转换为4x4变换矩阵"""
        if len(pose) != 6:
            raise ValueError("位姿必须是长度为6的列表：[x, y, z, rx, ry, rz]")
        x, y, z, rx, ry, rz = pose

        R = PointCloudGenerator.rotation_matrix(rx, ry, rz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def get_point_cloud_bounds(self, point_cloud: np.ndarray = None) -> dict:
        """获取点云X/Y/Z轴的最大值和最小值"""
        target_pcd = point_cloud if point_cloud is not None else self.generated_point_cloud

        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        x_min, y_min, z_min = np.min(target_pcd, axis=0)
        x_max, y_max, z_max = np.max(target_pcd, axis=0)

        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max
        }

    def get_point_cloud_center(self, point_cloud: np.ndarray = None, use_bounds: bool = True) -> np.ndarray:
        """计算点云的中心点坐标"""
        target_pcd = point_cloud if point_cloud is not None else self.generated_point_cloud

        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        if use_bounds:
            bounds = self.get_point_cloud_bounds(target_pcd)
            x_center = (bounds["x_min"] + bounds["x_max"]) / 2
            y_center = (bounds["y_min"] + bounds["y_max"]) / 2
            z_center = (bounds["z_min"] + bounds["z_max"]) / 2
        else:
            x_center, y_center, z_center = np.mean(target_pcd, axis=0)

        return np.array([x_center, y_center, z_center])

    @staticmethod
    def downsample_image(
            color_img: np.ndarray,
            depth_img: np.ndarray,
            scale_percent: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """对彩色图和深度图进行均匀下采样"""
        if scale_percent <= 0 or scale_percent > 1:
            raise ValueError("下采样比例必须在(0, 1]范围内")
        if scale_percent == 1.0:
            return color_img.copy(), depth_img.copy()

        interval = int(1 / np.sqrt(scale_percent))
        resized_color = np.zeros_like(color_img)
        resized_depth = np.zeros_like(depth_img)

        for i in range(0, color_img.shape[0], interval):
            for j in range(0, color_img.shape[1], interval):
                resized_color[i, j] = color_img[i, j]
                resized_depth[i, j] = depth_img[i, j]

        return resized_color, resized_depth

    def process_images(
            self,
            color_img: np.ndarray,
            depth_img: np.ndarray,
            color_scale: float = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """图像预处理（彩色图缩放 + 深度图中心裁剪）"""
        if color_img is None or depth_img is None:
            raise ValueError("彩色图或深度图为空，请检查输入")
        if color_img.shape[:2] != depth_img.shape[:2]:
            raise ValueError("原始彩色图与深度图的尺寸必须一致（H×W）")

        h_orig, w_orig = color_img.shape[:2]
        new_w = int(w_orig * color_scale)
        new_h = int(h_orig * color_scale)
        resized_color = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        start_x = (w_orig - new_w) // 2
        start_y = (h_orig - new_h) // 2
        cropped_depth = depth_img[start_y:start_y + new_h, start_x:start_x + new_w]

        return resized_color, cropped_depth

    def matrix_to_pose(self, matrix: np.ndarray) -> list:
        """将4x4变换矩阵转换为6D位姿 [x,y,z,rx,ry,rz]"""
        if matrix.shape != (4, 4):
            raise ValueError("输入必须是4x4的变换矩阵")

        x, y, z = matrix[:3, 3]
        R = matrix[:3, :3]

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0

        return [x, y, z, rx, ry, rz]

    def calculate_camera_pose(self) -> tuple[list, np.ndarray]:
        """
        计算相机在基坐标系下的位姿
        转换关系：相机在基坐标系位姿 = TCP在基坐标系位姿 × 相机相对于TCP的位姿
        """
        # 将TCP位姿转换为变换矩阵
        tcp_matrix = self.pose_to_matrix(self.tcp_pose)
        # 将相机相对于TCP的位姿转换为变换矩阵
        camera_tcp_matrix = self.pose_to_matrix(self.camera_to_tcp_pose)
        # 计算相机在基坐标系下的变换矩阵
        camera_matrix = np.dot(tcp_matrix, camera_tcp_matrix)
        # 转换为6D位姿
        camera_pose = self.matrix_to_pose(camera_matrix)

        return camera_pose, camera_matrix

    def apply_mask(
            self,
            color_img: np.ndarray,
            depth_img: np.ndarray,
            mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """应用掩码到彩色图和深度图"""
        if mask.shape[:2] != color_img.shape[:2]:
            mask = cv2.resize(mask, (color_img.shape[1], color_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)
        color_masked = color_img * np.stack([mask] * 3, axis=-1)
        depth_masked = depth_img * mask

        return color_masked, depth_masked

    def _visualization_loop(self, pcd, axis, window_name):
        """可视化循环，在单独线程中运行"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=window_name)
            self.vis.add_geometry(pcd)
            self.vis.add_geometry(axis)
            self.vis.run()
            self.vis.destroy_window()
            self.vis = None
        except Exception as e:
            print(f"可视化线程出错: {str(e)}")
            if self.vis:
                self.vis.destroy_window()
                self.vis = None

    def _safe_visualize(self, pcd, axis, window_name):
        """安全的可视化方法，使用线程避免阻塞主进程"""
        if self.visualization_thread and self.visualization_thread.is_alive():
            print("等待现有可视化窗口关闭...")
            self.visualization_thread.join(timeout=5.0)

            if self.visualization_thread.is_alive():
                print("强制终止现有可视化线程")
                if self.vis:
                    self.vis.destroy_window()
                self.visualization_thread = None

        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            args=(pcd, axis, window_name),
            daemon=True
        )
        self.visualization_thread.start()

    def generate_point_cloud(
            self,
            color_image_ori: np.ndarray,
            depth_image_ori: np.ndarray,
            mask: np.ndarray = None,
            downsample_scale: float = 1.0,
            image_alignment_scale: float = 1.0
    ) -> dict:
        """核心方法：生成点云并转换到基坐标系"""
        if color_image_ori is None:
            return {"state": "fail", "info": "原始彩色图为空", "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}
        if depth_image_ori is None:
            return {"state": "fail", "info": "原始深度图为空", "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}

        try:
            # 1. 图像预处理（对齐）
            color_img, depth_img = self.process_images(
                color_image_ori, depth_image_ori, color_scale=image_alignment_scale
            )

            # 2. 应用掩码（如果提供）
            processing_mode = "带掩码区域" if mask is not None else "完整"
            if mask is not None:
                color_img, depth_img = self.apply_mask(color_img, depth_img, mask)
                if not np.any(depth_img > 0):
                    return {"state": "fail", "info": "掩码区域内无有效深度数据",
                            "x": None, "y": None, "z": None,
                            "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}

            # 3. 图像下采样
            color_down, depth_down = self.downsample_image(
                color_img, depth_img, scale_percent=downsample_scale
            )

            # 4. 计算调整后的相机内参
            scale = downsample_scale * image_alignment_scale
            fx_f = self.fx * scale
            fy_f = self.fy * scale
            cx_f = self.cx * scale
            cy_f = self.cy * scale

            # 5. 生成相机坐标系下的点云
            valid_mask = depth_down > 0
            if not np.any(valid_mask):
                return {"state": "fail", "info": f"{processing_mode}图像无有效深度数据",
                        "x": None, "y": None, "z": None,
                        "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}

            valid_depth = depth_down[valid_mask]
            valid_color = color_down[valid_mask]
            v, u = np.where(valid_mask)

            x_cam = (u - cx_f) * valid_depth / fx_f
            y_cam = (v - cy_f) * valid_depth / fy_f
            z_cam = valid_depth
            points_cam = np.stack((x_cam, y_cam, z_cam), axis=1)

            # 6. 坐标转换：相机坐标系 → 基坐标系
            # 6.1 先转换到TCP坐标系（使用相机相对于TCP的位姿）
            camera_to_tcp_matrix = self.pose_to_matrix(self.camera_to_tcp_pose)
            R_cam_tcp = camera_to_tcp_matrix[:3, :3]  # 相机到TCP的旋转矩阵
            T_cam_tcp = camera_to_tcp_matrix[:3, 3]  # 相机到TCP的平移向量
            points_tcp = np.dot(points_cam, R_cam_tcp.T) + T_cam_tcp

            # 6.2 再转换到基坐标系（使用TCP在基坐标系的位姿）
            tcp_to_base_matrix = self.pose_to_matrix(self.tcp_pose)
            R_tcp_base = tcp_to_base_matrix[:3, :3]  # TCP到基坐标系的旋转矩阵
            T_tcp_base = tcp_to_base_matrix[:3, 3]  # TCP到基坐标系的平移向量
            points_world = np.dot(points_tcp, R_tcp_base.T) + T_tcp_base

            # 7. 缓存生成的点云
            self.generated_point_cloud = points_world

            # 8. 计算点云信息
            bounds = self.get_point_cloud_bounds()
            center = self.get_point_cloud_center()
            hx, hy, hz = center[0], center[1], center[2]
            x_min, y_min, z_min = bounds["x_min"], bounds["y_min"], bounds["z_min"]
            x_max, y_max, z_max = bounds["x_max"], bounds["y_max"], bounds["z_max"]

            # 9. 保存点云
            if self.save_point_cloud:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                np.save(self.save_path, points_world)
                print(f"点云已保存至: {self.save_path}")

            # 10. 可视化点云
            if self.visualize and len(points_world) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_world)
                colors = valid_color / 255.0
                colors = colors[:, [2, 1, 0]]  # BGR转RGB
                pcd.colors = o3d.utility.Vector3dVector(colors)
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)

                self._safe_visualize(pcd, axis, f"{processing_mode}点云可视化")
                print("可视化窗口已打开，关闭窗口后程序将继续执行...")

            # 11. 返回结果
            return {
                "state": "success",
                "info": f"{processing_mode}点云生成成功，包含{len(points_world)}个点",
                "x": hx, "y": hy, "z": hz,
                "x_min": x_min, "y_min": y_min, "z_min": z_min,
                "x_max": x_max, "y_max": y_max, "z_max": z_max,
                "point_count": len(points_world),
                "point_cloud": points_world
            }

        except Exception as e:
            self.generated_point_cloud = None
            if self.vis:
                self.vis.destroy_window()
                self.vis = None
            if self.visualization_thread and self.visualization_thread.is_alive():
                self.visualization_thread.join(timeout=2.0)
            return {"state": "fail", "info": f"点云生成失败：{str(e)}",
                    "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0,
                    "point_count": 0}


# 测试代码
if __name__ == "__main__":
    # 1. 相机内参
    fx, fy, cx, cy = 604.95, 604.95, 316.23, 233.86  # cam1
    # fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67  # cam2

    # 2. 定义相机相对于TCP的6D位姿（示例）
    # 含义：相机在TCP坐标系中位于(0, 0, 150mm)，绕X轴旋转90度
    camera_to_tcp_pose = [30, -100, 30, 0 / 180 * pi, 0 / 180 * pi, -180 / 180 * pi]

    # 3. TCP在基坐标系中的6D位姿（示例）
    # tcp_pose = [315.109, 306.038, 607.385, 180.0 / 180 * pi, 0 / 180 * pi, -50 / 180 * pi]
    # tcp_pose = [201.81, -316.67, 942.86, -126.17 / 180 * pi, -0.21 / 180 * pi,  -130.62 / 180 * pi]

    tcp_pose = [  225.16, -352.58, 649.92, --89.10 / 180 * pi,  0.42 / 180 * pi, -140.07 / 180 * pi]
    # 4. 读取图像
    def read_image_safely(path, is_depth=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"图像文件不存在：{path}")
        if is_depth:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(path)
        if img is None:
            raise ValueError(f"图像读取失败：{path}")
        return img


    try:
        color_path = "output/new/color_2025-11-15_190340_606.png"
        depth_path = "output/new/depth_16bit_2025-11-15_190340_606.png"
        mask_path = "C:\\Users\\lenovo\\Desktop\\robot_801\\output\\cam1_single\\res_20251115_190344\\masks\\mask_id_1_label_a_pink_block_conf_0.71.png"

        color_img = read_image_safely(color_path, is_depth=False)
        depth_img = read_image_safely(depth_path, is_depth=True)
        mask = read_image_safely(mask_path, is_depth=True)
        print("图像读取成功！")
    except Exception as e:
        print(f"图像读取失败：{str(e)}")
        exit()

    # 5. 创建点云生成器实例（传入6D位姿）
    pcd_generator = PointCloudGenerator(
        fx=fx, fy=fy, cx=cx, cy=cy,
        tcp_pose=tcp_pose,
        camera_to_tcp_pose=camera_to_tcp_pose,  # 传入相机相对于TCP的6D位姿
        visualize=True,
        save_point_cloud=False
    )

    # 6. 生成带掩码的点云
    print("\n=== 测试1：带掩码模式 ===")
    mask_response = pcd_generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask=mask,
        downsample_scale=1
    )

    point_cloud_with_mask = mask_response["point_cloud"]
    np.save('point_cloud_with_mask.npy', point_cloud_with_mask)

    if pcd_generator.visualization_thread:
        pcd_generator.visualization_thread.join()

    if mask_response["state"] == "success":
        print(f"掩码点云结果：{mask_response['info']}")
        print(f"中心点：({mask_response['x']:.2f}, {mask_response['y']:.2f}, {mask_response['z']:.2f})")

    # 7. 生成完整点云
    print("\n=== 测试2：完整点云模式 ===")
    full_response = pcd_generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask=None,
        downsample_scale=1
    )

    if pcd_generator.visualization_thread:
        pcd_generator.visualization_thread.join()

    if full_response["state"] == "success":
        print(f"完整点云结果：{full_response['info']}")
        print(f"中心点：({full_response['x']:.2f}, {full_response['y']:.2f}, {full_response['z']:.2f})")

    print("\n=== 所有测试完成 ===")
