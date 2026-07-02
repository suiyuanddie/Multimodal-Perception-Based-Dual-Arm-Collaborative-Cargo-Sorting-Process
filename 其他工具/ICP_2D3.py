import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import traceback

# ------------------------------ 全局配置 ------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 外部点云默认配置（用户可修改）
DEFAULT_EXTERNAL_CLOUD_PATH = "input_cloud.ply"  # 你的点云文件路径（支持PLY/PCD等格式）
DEFAULT_CLOUD_UNIT = "meter"  # 输入点云单位（"meter"米 / "millimeter"毫米）
DEFAULT_CUBE_SIDE = 0.1  # 实际立方体边长（米）- 必须与点云物体一致！
DEFAULT_REAL_YAW = None  # 若已知点云真实Yaw角（用于误差验证），可填入（如60.0）


# ------------------------------ 外部点云读取工具 ------------------------------
def read_external_point_cloud(
        file_path: str,
        unit: str = "meter"
) -> np.ndarray:
    """
    读取外部3D点云文件，转换为毫米单位的N×3 numpy数组
    支持格式：PLY、PCD、XYZ等Open3D兼容格式

    :param file_path: 点云文件路径（相对/绝对路径）
    :param unit: 输入点云的原始单位，可选"meter"（米）或"millimeter"（毫米）
    :return: 毫米单位的点云数组 (N, 3)，每一行对应一个点的(X,Y,Z)坐标
    """
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"点云读取失败！文件「{file_path}」为空或格式不支持")

    # 转换为numpy数组并验证维度
    cloud_np = np.asarray(pcd.points)
    if cloud_np.shape[1] != 3:
        raise ValueError(f"点云维度错误！需为3D点云（N×3），当前格式：{cloud_np.shape}")

    # 单位统一转换为毫米（原代码核心单位）
    if unit.lower() == "meter":
        cloud_mm = cloud_np * 1000.0  # 米 → 毫米
    elif unit.lower() == "millimeter":
        cloud_mm = cloud_np  # 已为毫米，直接使用
    else:
        raise ValueError(f"不支持的单位「{unit}」！请选择 'meter' 或 'millimeter'")

    print(f"[外部点云读取完成]")
    print(f"  - 文件路径：{file_path}")
    print(f"  - 点云点数：{len(cloud_mm)}")
    print(f"  - 原始单位：{unit} → 转换后单位：毫米")
    print(f"  - 坐标范围：X[{cloud_mm[:, 0].min():.1f}, {cloud_mm[:, 0].max():.1f}]mm")
    print(f"            Y[{cloud_mm[:, 1].min():.1f}, {cloud_mm[:, 1].max():.1f}]mm")
    print(f"            Z[{cloud_mm[:, 2].min():.1f}, {cloud_mm[:, 2].max():.1f}]mm")
    return cloud_mm


# ------------------------------ 核心位姿估计类（适配外部点云） ------------------------------
class CubePoseEstimator:
    def __init__(self,
                 cube_side: float = 0.1,  # 实际立方体边长（米）- 必须与物体一致
                 save_result: bool = True,
                 visualize: bool = True):
        self.cube_side = cube_side
        self.save_result = save_result
        self.visualize = visualize
        self.real_yaw = None  # 外部点云真实Yaw角（可选，用于误差验证）

        self.target_points = None  # 目标点云：外部点云提取的上表面（XY坐标，米单位）
        self.source_points = None  # 源点云：生成的标准方形点云（0°初始姿态）
        self.object_center_xy = np.array([0.0, 0.0])  # 物体中心（上表面均值，米单位）

        # 结果存储字典
        self.pose_result = {
            "yaw_angle": 0.0,  # 最终旋转姿态（Yaw角，逆时针为正，°）
            "yaw_error": None,  # 与真实Yaw角的误差（°，仅当real_yaw已知时有效）
            "matching_rmse": 0.0,  # 点云匹配误差（毫米）
            "translation": [0.0, 0.0],  # 平移量（毫米）
            "object_center_xy": [0.0, 0.0],  # 物体中心（毫米）
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "save_dir": None,
            "cube_side": cube_side * 1000  # 立方体边长（毫米）
        }

        # 初始化结果保存目录
        if self.save_result:
            self.save_dir = Path(f"output/ICP_result/{self.pose_result['timestamp']}")
            self.save_dir.mkdir(exist_ok=True, parents=True)
            self.pose_result["save_dir"] = str(self.save_dir)
        else:
            self.save_dir = None

    def load_point_cloud(
            self,
            point_cloud_mm: np.ndarray,  # 外部点云（毫米单位，N×3）
            real_yaw: float = None,  # 真实Yaw角（可选）
            z_tolerance: float = 0.0005  # 上表面Z值筛选公差（米，默认0.5毫米）
    ):
        """
        加载外部点云，提取上表面点（核心步骤）
        :param point_cloud_mm: 外部点云数组（毫米单位，N×3）
        :param real_yaw: 真实Yaw角（可选，用于误差验证）
        :param z_tolerance: 上表面Z值公差（米），噪声大时可适当增大（如0.001=1毫米）
        """
        # 输入合法性检查
        if not (isinstance(point_cloud_mm, np.ndarray) and point_cloud_mm.shape[1] == 3):
            raise ValueError("点云格式错误！需为(N×3)的numpy数组（X,Y,Z，毫米单位）")
        if len(point_cloud_mm) < 100:
            raise ValueError(f"点云点数不足（{len(point_cloud_mm)}点），至少需100点")

        # 转换为米单位（便于后续计算）
        point_cloud_m = point_cloud_mm / 1000.0
        z_values = point_cloud_m[:, 2]

        # -------------------------- 核心：提取上表面 --------------------------
        # 上表面特征：Z值最大（假设点云Z轴垂直向上）
        max_z = np.max(z_values)  # 上表面理论Z值（米）
        z_threshold = max_z - z_tolerance  # 筛选阈值（保留Z接近最大值的点）
        surface_mask = z_values >= z_threshold  # 上表面点掩码
        self.target_points = point_cloud_m[surface_mask, :2]  # 仅保留XY坐标（目标点云）

        # 上表面点数检查
        if len(self.target_points) < 50:
            raise RuntimeError(
                f"上表面提取失败！仅提取到{len(self.target_points)}点（需≥50点）\n"
                f"建议：1. 检查点云Z轴方向是否垂直向上 2. 增大z_tolerance（当前{z_tolerance * 1000}毫米）"
            )

        # 计算物体中心（上表面点的XY均值）
        self.object_center_xy = np.mean(self.target_points, axis=0)
        self.pose_result["object_center_xy"] = (self.object_center_xy * 1000).tolist()  # 转毫米
        self.real_yaw = real_yaw

        # 生成标准方形点云（源点云，与上表面点数一致，0°初始姿态）
        self._create_standard_source(preset_rot=0.0)

        # 打印加载信息
        print(f"\n[点云加载与上表面提取完成]")
        print(f"  - 上表面点数：{len(self.target_points)}（原始点云{len(point_cloud_mm)}点）")
        print(
            f"  - 物体中心（毫米）：({self.pose_result['object_center_xy'][0]:.1f}, {self.pose_result['object_center_xy'][1]:.1f})")
        print(f"  - 上表面Z值范围（米）：{z_threshold:.6f} ~ {max_z:.6f}（公差{z_tolerance * 1000}毫米）")
        if self.real_yaw is not None:
            print(f"  - 真实Yaw角（参考）：{self.real_yaw:.1f}°")
        return self

    def _create_standard_source(self, preset_rot: float = 0.0):
        """
        生成标准方形点云（源点云）：与上表面同尺寸、同中心，按预设角度旋转
        :param preset_rot: 预设旋转角（°，逆时针为正）
        """
        half_side = self.cube_side / 2  # 立方体半边长（米）
        n = len(self.target_points)  # 与上表面点数一致，保证匹配公平性

        # 生成局部正方形点云（-half_side ~ half_side，米单位）
        x_local = np.random.uniform(-half_side, half_side, n)
        y_local = np.random.uniform(-half_side, half_side, n)
        xy_local = np.vstack([x_local, y_local]).T

        # 按预设角度旋转 + 平移到物体中心
        rot_rad = np.radians(preset_rot)
        rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                            [np.sin(rot_rad), np.cos(rot_rad)]])
        xy_rot_local = xy_local @ rot_mat
        self.source_points = xy_rot_local + self.object_center_xy  # 最终源点云（米单位）

        print(f"[标准方形点云生成] 点数：{n} | 初始旋转角：{preset_rot:.1f}°（边与XY轴平行）")
        return self

    def _find_initial_yaw(self):
        """
        搜索初始Yaw角：通过全局粗搜+重点区域精搜，找到接近真实值的初始姿态（防止ICP发散）
        """
        # 构建目标点云KD树（加速最近邻搜索）
        target_pcd = o3d.geometry.PointCloud()
        target_3d = np.hstack([self.target_points, np.zeros((len(self.target_points), 1))])  # 补Z=0
        target_pcd.points = o3d.utility.Vector3dVector(target_3d)
        target_kd = o3d.geometry.KDTreeFlann(target_pcd)

        best_error = float('inf')
        best_yaw = 0.0
        total_points = len(self.source_points)

        # 1. 定义搜索范围：真实角±45°（若已知）+ 全局0~360°
        search_angles = list(range(0, 360, 2))  # 全局粗搜（步长2°）
        if self.real_yaw is not None:
            # 重点区域精搜（步长0.2°）
            expected_range = [(self.real_yaw - 45) % 360, (self.real_yaw + 45) % 360]
            if expected_range[0] < expected_range[1]:
                focus_angles = np.arange(expected_range[0], expected_range[1], 0.2)
            else:  # 跨360°边界（如350°~10°）
                focus_angles = np.hstack([np.arange(expected_range[0], 360, 0.2),
                                          np.arange(0, expected_range[1], 0.2)])
            search_angles.extend(focus_angles.tolist())
        search_angles = sorted(list(set(search_angles)))  # 去重排序

        # 2. 遍历所有角度，筛选最优初始角
        print(f"\n[初始Yaw角搜索] 共搜索{len(search_angles)}个角度...")
        for yaw in search_angles:
            # 旋转源点云
            rot_rad = np.radians(yaw)
            rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                                [np.sin(rot_rad), np.cos(rot_rad)]])
            rotated_source = (self.source_points - self.object_center_xy) @ rot_mat + self.object_center_xy

            # 筛选有效对应点（距离<10毫米，过滤噪声）
            corrs_dist = []
            for p in rotated_source:
                _, idx, dist = target_kd.search_knn_vector_3d(np.array([p[0], p[1], 0.0]), 1)
                if dist[0] < 0.01:  # 10毫米（米单位）
                    corrs_dist.append(dist[0])

            # 有效条件：对应点比例≥70% + 点数≥50
            if len(corrs_dist) / total_points >= 0.7 and len(corrs_dist) >= 50:
                rmse = np.sqrt(np.mean(np.square(corrs_dist)))
                if rmse < best_error:
                    best_error = rmse
                    best_yaw = yaw

        # 降级策略：若未找到高比例对应点，降低阈值到50%
        if best_error == float('inf'):
            print(f"[警告] 高比例对应点未找到，降低阈值至50%...")
            for yaw in search_angles:
                rot_rad = np.radians(yaw)
                rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                                    [np.sin(rot_rad), np.cos(rot_rad)]])
                rotated_source = (self.source_points - self.object_center_xy) @ rot_mat + self.object_center_xy

                corrs_dist = []
                for p in rotated_source:
                    _, idx, dist = target_kd.search_knn_vector_3d(np.array([p[0], p[1], 0.0]), 1)
                    if dist[0] < 0.01:
                        corrs_dist.append(dist[0])

                if len(corrs_dist) / total_points >= 0.5 and len(corrs_dist) >= 50:
                    rmse = np.sqrt(np.mean(np.square(corrs_dist)))
                    if rmse < best_error:
                        best_error = rmse
                        best_yaw = yaw

        # 最终检查：若仍无有效角度，报错
        if best_error == float('inf'):
            raise RuntimeError("初始Yaw角搜索失败！请检查点云质量或调整搜索参数")

        print(f"[初始Yaw角搜索完成] 最优初始角：{best_yaw:.1f}° | RMSE：{best_error * 1000:.2f}mm")
        return best_yaw

    def _icp_matching(self, initial_yaw: float):
        """
        ICP精修：基于初始角优化旋转和平移，实现高精度匹配
        :param initial_yaw: 初始Yaw角（°）
        """
        # 构建目标点云KD树
        target_pcd = o3d.geometry.PointCloud()
        target_3d = np.hstack([self.target_points, np.zeros((len(self.target_points), 1))])
        target_pcd.points = o3d.utility.Vector3dVector(target_3d)
        target_kd = o3d.geometry.KDTreeFlann(target_pcd)

        # 初始化旋转矩阵和平移向量
        rot_rad = np.radians(initial_yaw)
        current_rot = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                                [np.sin(rot_rad), np.cos(rot_rad)]])
        current_trans = np.array([0.0, 0.0])  # 初始平移为0

        # ICP迭代参数
        max_iter = 200
        tolerance = 1e-8  # RMSE变化阈值（收敛条件）
        prev_rmse = float('inf')
        rmse_history = []  # 记录RMSE，防止发散

        print(f"\n[ICP精修] 开始迭代（最大{max_iter}次）...")
        for iter_idx in range(max_iter):
            # 1. 应用当前旋转和平移到源点云
            rotated_source = (
                                         self.source_points - self.object_center_xy) @ current_rot + self.object_center_xy + current_trans

            # 2. 筛选有效对应点（距离<10毫米）
            src_corrs = []
            tgt_corrs = []
            for p in rotated_source:
                _, idx, dist = target_kd.search_knn_vector_3d(np.array([p[0], p[1], 0.0]), 1)
                if dist[0] < 0.01:  # 10毫米（米单位）
                    src_corrs.append(p)
                    tgt_corrs.append(self.target_points[idx[0]])
            if len(src_corrs) < 50:
                print(f"[警告] 有效对应点不足（{len(src_corrs)}点），提前停止迭代")
                break
            src_corrs = np.array(src_corrs)
            tgt_corrs = np.array(tgt_corrs)

            # 3. SVD求解最优旋转和平移（最小二乘意义）
            src_mean = np.mean(src_corrs, axis=0)
            tgt_mean = np.mean(tgt_corrs, axis=0)
            src_centered = src_corrs - src_mean
            tgt_centered = tgt_corrs - tgt_mean

            # 计算协方差矩阵
            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            rot_update = Vt.T @ U.T

            # 确保旋转矩阵为正定（避免镜像翻转）
            if np.linalg.det(rot_update) < 0:
                Vt[-1, :] *= -1
                rot_update = Vt.T @ U.T

            # 计算平移更新
            trans_update = tgt_mean - src_mean @ rot_update

            # 4. 更新旋转和平移
            current_rot = rot_update @ current_rot
            current_trans = rot_update @ current_trans + trans_update

            # 5. 计算当前RMSE并检查收敛
            src_transformed = src_corrs @ rot_update + trans_update
            current_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(src_transformed - tgt_corrs, axis=1))))
            rmse_history.append(current_rmse)

            # 保留最近3次RMSE，防止发散
            if len(rmse_history) > 3:
                rmse_history.pop(0)

            # 收敛条件1：RMSE变化小于阈值
            if abs(prev_rmse - current_rmse) < tolerance:
                print(f"[ICP精修] 迭代{iter_idx + 1}次收敛 | RMSE：{current_rmse * 1000:.2f}mm")
                break

            # 收敛条件2：RMSE连续3次上升（发散），停止迭代
            if len(rmse_history) == 3 and all(rmse_history[i] < rmse_history[i + 1] for i in range(2)):
                print(f"[ICP精修] 迭代{iter_idx + 1}次停止（RMSE发散） | RMSE：{current_rmse * 1000:.2f}mm")
                break

            prev_rmse = current_rmse
            # 每50次迭代打印进度
            if (iter_idx + 1) % 50 == 0:
                print(f"[ICP精修] 迭代{iter_idx + 1}次 | RMSE：{current_rmse * 1000:.2f}mm")
        else:
            print(f"[ICP精修] 达到最大迭代次数 | RMSE：{current_rmse * 1000:.2f}mm")

        # 计算最终Yaw角（从旋转矩阵提取）
        final_yaw = np.degrees(np.arctan2(current_rot[1, 0], current_rot[0, 0])) % 360  # 归一化到0~360°

        # 更新结果字典
        self.pose_result["yaw_angle"] = final_yaw
        self.pose_result["matching_rmse"] = current_rmse * 1000  # 转毫米
        self.pose_result["translation"] = (current_trans * 1000).tolist()  # 转毫米
        if self.real_yaw is not None:
            self.pose_result["yaw_error"] = abs(final_yaw - self.real_yaw)

        print(f"[ICP精修结果] 估计Yaw角：{final_yaw:.2f}° | 匹配误差：{self.pose_result['matching_rmse']:.2f}mm")
        return self

    def _resolve_ambiguity(self):
        """
        解决正方形旋转歧义：正方形存在90°倍数旋转对称性，需筛选最优解
        """
        print(f"\n[歧义解决] 处理正方形90°旋转对称性...")
        current_yaw = self.pose_result["yaw_angle"]
        # 正方形的歧义角（90°倍数偏移）
        ambiguity_offsets = [0, 90, 180, 270, -90, -180, -270]
        candidates = [(current_yaw + off) % 360 for off in ambiguity_offsets]

        # 若已知真实Yaw角，加入其歧义角（提高正确概率）
        if self.real_yaw is not None:
            real_candidates = [(self.real_yaw + off) % 360 for off in ambiguity_offsets]
            candidates.extend(real_candidates)
        candidates = sorted(list(set(candidates)))  # 去重排序
        print(f"  候选Yaw角：{[round(y, 1) for y in candidates]}°")

        # 构建目标点云KD树
        target_pcd = o3d.geometry.PointCloud()
        target_3d = np.hstack([self.target_points, np.zeros((len(self.target_points), 1))])
        target_pcd.points = o3d.utility.Vector3dVector(target_3d)
        target_kd = o3d.geometry.KDTreeFlann(target_pcd)

        best_score = float('inf')
        best_yaw = current_yaw
        total_points = len(self.source_points)
        half_side = self.cube_side / 2  # 半边长（米）

        # 遍历候选角，计算综合得分（得分越低越优）
        for yaw_cand in candidates:
            # 旋转源点云（候选角）
            rot_rad = np.radians(yaw_cand)
            rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                                [np.sin(rot_rad), np.cos(rot_rad)]])

            # 同步旋转平移向量
            trans_m = np.array(self.pose_result["translation"]) / 1000  # 转米
            rotated_trans = trans_m @ rot_mat.T

            # 应用旋转和平移
            rotated_source = (
                                         self.source_points - self.object_center_xy) @ rot_mat + self.object_center_xy + rotated_trans

            # 1. 形状匹配度：正方形X/Y范围差异（越小越优）
            x_range = np.max(rotated_source[:, 0]) - np.min(rotated_source[:, 0])
            y_range = np.max(rotated_source[:, 1]) - np.min(rotated_source[:, 1])
            range_diff = abs(x_range - y_range)

            # 2. 尺寸匹配度：与实际边长的差异（越小越优）
            avg_size = (x_range + y_range) / 2
            size_diff = abs(avg_size - (2 * half_side))  # 理论边长：2*半边长

            # 3. 点云匹配度：对应点平均距离（越小越优）
            dist_errors = []
            for p in rotated_source:
                _, idx, dist = target_kd.search_knn_vector_3d(np.array([p[0], p[1], 0.0]), 1)
                if dist[0] < 0.01:  # 10毫米（米单位）
                    dist_errors.append(dist[0])
            corr_ratio = len(dist_errors) / total_points
            # 低对应点惩罚（避免无效候选）
            dist_score = np.mean(dist_errors) if (corr_ratio >= 0.6 and dist_errors) else 0.01

            # 综合得分（权重：匹配度90% > 尺寸10% > 形状5%，优先保证对齐）
            base_score = range_diff * 0.05 + size_diff * 0.1 + dist_score * 0.9

            # 真实角距离惩罚（若已知）：优先选择接近真实角的候选
            angle_penalty = 0.0
            if self.real_yaw is not None:
                angle_diff = abs(yaw_cand - self.real_yaw)
                angle_diff = min(angle_diff, 360 - angle_diff)  # 最小角度差（0~180°）
                angle_penalty = angle_diff * 0.00001  # 每差1°增加0.00001得分

            total_score = base_score + angle_penalty
            print(f"  候选{yaw_cand:.1f}°：匹配得分{dist_score:.6f} | 总得分{total_score:.6f}")

            # 更新最优候选
            if total_score < best_score:
                best_score = total_score
                best_yaw = yaw_cand

        # 更新最终Yaw角
        self.pose_result["yaw_angle"] = best_yaw
        if self.real_yaw is not None:
            self.pose_result["yaw_error"] = abs(best_yaw - self.real_yaw)
            print(f"[歧义解决结果] 最终Yaw角：{best_yaw:.2f}° | 误差：{self.pose_result['yaw_error']:.2f}°（若已知真实角）")
        else:
            print(f"[歧义解决结果] 最终Yaw角：{best_yaw:.2f}°（未提供真实值，无误差）")
        return self

    def _save_result(self):
        """保存结果到JSON文件和可视化图像"""
        if not self.save_result:
            return

        # 序列化结果（兼容numpy类型）
        result_serializable = {}
        for k, v in self.pose_result.items():
            if isinstance(v, np.ndarray):
                result_serializable[k] = v.tolist()
            else:
                result_serializable[k] = v

        # 保存JSON结果
        json_path = self.save_dir / "pose_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_serializable, f, indent=4, ensure_ascii=False)
        print(f"\n[结果保存] JSON文件：{json_path}")

        # 保存可视化图像
        self.visualize_result(save_only=True)
        return self

    def visualize_result(self, save_only: bool = False):
        """
        可视化点云匹配结果
        :param save_only: 仅保存图像，不显示窗口（用于批量处理）
        """
        if self.target_points is None or self.source_points is None:
            raise RuntimeError("请先执行estimate_pose()完成位姿估计！")

        # 转换为毫米单位，用于可视化
        center_mm = np.array(self.pose_result["object_center_xy"])
        half_side_mm = self.pose_result["cube_side"] / 2

        # 1. 黄色点云：外部点云提取的上表面（目标点云）
        yellow_3d = np.hstack([self.target_points * 1000, np.zeros((len(self.target_points), 1))])
        yellow_pcd = o3d.geometry.PointCloud()
        yellow_pcd.points = o3d.utility.Vector3dVector(yellow_3d)
        yellow_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色

        # 2. 绿色点云：标准方形点云（初始0°，未旋转）
        green_3d = np.hstack([self.source_points * 1000, np.zeros((len(self.source_points), 1))])
        green_pcd = o3d.geometry.PointCloud()
        green_pcd.points = o3d.utility.Vector3dVector(green_3d)
        green_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色

        # 3. 蓝色点云：ICP对齐后的点云（结果）
        final_rot = np.array(
            [[np.cos(np.radians(self.pose_result["yaw_angle"])), -np.sin(np.radians(self.pose_result["yaw_angle"]))],
             [np.sin(np.radians(self.pose_result["yaw_angle"])), np.cos(np.radians(self.pose_result["yaw_angle"]))]])
        trans_mm = np.array(self.pose_result["translation"])
        blue_xy = (self.source_points - self.object_center_xy) @ final_rot + self.object_center_xy
        blue_3d = np.hstack([blue_xy * 1000 + trans_mm, np.zeros((len(blue_xy), 1))])
        blue_pcd = o3d.geometry.PointCloud()
        blue_pcd.points = o3d.utility.Vector3dVector(blue_3d)
        blue_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色

        # 坐标系：物体中心为原点，尺寸=立方体边长
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=half_side_mm * 1.5,  # 坐标系大小
            origin=[center_mm[0], center_mm[1], 0.0]  # 原点在物体中心（Z=0）
        )

        # 可视化窗口配置
        window_title = (
            f"点云匹配结果 | Yaw：{self.pose_result['yaw_angle']:.2f}° | RMSE：{self.pose_result['matching_rmse']:.2f}mm\n"
            f"黄=外部上表面 | 绿=标准0° | 蓝=对齐后"
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_title, width=1200, height=800)

        # 添加点云和坐标系（顺序：黄色在下，蓝色中间，绿色在上）
        vis.add_geometry(yellow_pcd)
        vis.add_geometry(blue_pcd)
        vis.add_geometry(green_pcd)
        vis.add_geometry(coord_frame)

        # 渲染参数（白色背景，点大小适中）
        render_opt = vis.get_render_option()
        render_opt.point_size = 8
        render_opt.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景

        # 视角配置（斜后方视角，便于观察角度）
        view_control = vis.get_view_control()
        view_control.set_lookat([center_mm[0], center_mm[1], 0.0])
        view_control.set_up([0.0, 0.0, 1.0])  # Z轴向上
        view_control.set_front([-1.5, -1.5, -0.8])  # 斜后方视角
        view_control.set_zoom(0.4)

        # 保存图像
        if self.save_result:
            img_path = self.save_dir / "matching_result.png"
            vis.capture_screen_image(str(img_path))
            print(f"[结果保存] 可视化图像：{img_path}")

        # 显示窗口（若不是仅保存）
        if not save_only:
            print("\n[可视化操作说明]")
            print("  - 左键拖动：旋转视角")
            print("  - 滚轮：缩放视角")
            print("  - 右键拖动：平移视角")
            print("  - 关闭窗口：继续程序")
            vis.run()
        vis.destroy_window()
        return self

    def estimate_pose(self):
        """完整位姿估计流程：初始角搜索 → ICP精修 → 歧义解决 → 结果保存"""
        if self.target_points is None:
            raise RuntimeError("请先调用load_point_cloud()加载外部点云！")

        # 打印流程标题
        print("\n" + "=" * 70)
        print(f"开始外部点云旋转姿态估计 | 时间：{self.pose_result['timestamp']}")
        print(f"立方体边长：{self.cube_side * 1000:.1f}mm | 结果保存：{self.save_result} | 可视化：{self.visualize}")
        print("=" * 70)

        # 核心流程
        initial_yaw = self._find_initial_yaw()  # 1. 初始角搜索
        self._icp_matching(initial_yaw)  # 2. ICP精修
        self._resolve_ambiguity()  # 3. 歧义解决

        # 保存结果
        if self.save_result:
            self._save_result()

        # 可视化
        if self.visualize:
            self.visualize_result()

        # 打印最终结果
        print("\n" + "=" * 70)
        print("外部点云旋转姿态估计最终结果")
        print("=" * 70)
        print(f"1. 旋转姿态（Yaw角）：{self.pose_result['yaw_angle']:.2f}°（逆时针为正）")
        if self.pose_result["yaw_error"] is not None:
            print(f"2. Yaw角误差（与真实值）：{self.pose_result['yaw_error']:.2f}°")
        else:
            print("2. Yaw角误差：未提供真实Yaw角，无法计算")
        print(f"3. 点云匹配误差：{self.pose_result['matching_rmse']:.2f}mm")
        print(
            f"4. 物体中心（XY）：({self.pose_result['object_center_xy'][0]:.1f}, {self.pose_result['object_center_xy'][1]:.1f})mm")
        print(f"5. 平移补偿：({self.pose_result['translation'][0]:.2f}, {self.pose_result['translation'][1]:.2f})mm")
        if self.save_result:
            print(f"6. 结果保存目录：{self.pose_result['save_dir']}")
        print("=" * 70)

        return self.pose_result


# ------------------------------ 测试入口（用户可直接运行） ------------------------------
def test_external_point_cloud_pose():
    """测试外部点云姿态估计流程"""
    # -------------------------- 用户配置（关键！请根据实际情况修改） --------------------------
    external_cloud_path = DEFAULT_EXTERNAL_CLOUD_PATH  # 你的点云文件路径
    external_cloud_unit = DEFAULT_CLOUD_UNIT  # 点云单位（"meter" / "millimeter"）
    cube_side = DEFAULT_CUBE_SIDE  # 立方体实际边长（米）
    real_yaw = DEFAULT_REAL_YAW  # 真实Yaw角（可选，用于误差验证）
    z_tolerance = 0.0005  # 上表面Z值公差（米，噪声大时调大）
    # ----------------------------------------------------------------------------------------

    try:
        # 1. 读取外部点云（毫米单位）
        external_cloud_mm = read_external_point_cloud(
            file_path=external_cloud_path,
            unit=external_cloud_unit
        )

        # 2. 初始化姿态估计器并执行估计
        estimator = CubePoseEstimator(
            cube_side=cube_side,
            save_result=True,  # 保存结果到文件
            visualize=True  # 显示可视化窗口
        )
        result = estimator.load_point_cloud(
            point_cloud_mm=external_cloud_mm,
            real_yaw=real_yaw,
            z_tolerance=z_tolerance
        ).estimate_pose()

        # 3. 结果验证（可选）
        if result["matching_rmse"] > 10.0:
            print(f"\n[警告] 匹配误差较大（{result['matching_rmse']:.2f}mm），建议检查：")
            print("  1. 点云是否为立方体上表面（Z轴垂直向上）")
            print("  2. cube_side是否与实际物体边长一致")
            print("  3. z_tolerance是否足够（噪声大时需增大）")

        return result

    except Exception as e:
        print(f"\n[姿态估计失败] 原因：{type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None


# ------------------------------ 运行入口 ------------------------------
if __name__ == "__main__":
    # 运行外部点云姿态估计
    test_external_point_cloud_pose()