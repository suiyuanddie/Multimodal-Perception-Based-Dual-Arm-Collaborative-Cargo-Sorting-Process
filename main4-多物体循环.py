import math
import sys
import os
import time
from WooshWebSocketClient import WooshApi#调底盘
from use_http_arm1_robot import RobotAPIClient, GripperAPIClient  # 专用arm1和夹爪1
from use_http_arm2_robot import RobotController
from use_http_camera import D435CameraClient#调相机
from use_http_groundedSAM import call_grounded_sam_service#掉SAM
from use_http_gripper2 import GripperClient  # 专用夹爪2
from math import pi,sin,cos
from pointcloud import PointCloudGenerator  # 点云生成服务
import numpy as np
import cv2
from ICP_2D3 import CubePoseEstimator
def normalize_arm1_position(robot_client: RobotAPIClient) -> bool:#标准化移动机械臂姿态
    try:
        print("===  机器人1控制 ===")
        print(f"连接状态：{robot_client.get_status()}")
        print(f"当前TCP位姿：{robot_client.get_tcp_pose()}")
        # 标准坐标
        target_pose = {
            "x": -129.0, "y": -12.0, "z": 915.0,
            "r": -89.0, "p": -1.0, "_y_": -135.0
        }
        ptp_result = robot_client.run_point(
            pose=target_pose, speed=30, tolerance={"x": 1.5, "y": 1.5}
        )
        print(ptp_result)
        print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
        print(f"运动误差：{ptp_result['verification']['errors']}")
    except Exception as e:
        print(f"机器人控制异常：{str(e)}")
    finally:
        pass
        # print(f"断开机器人连接：{robot_client.disconnect()}")
class Pose:
        """机械臂末端TCP位姿结构体"""
        def __init__(self, x=0.0, y=0.0, z=0.0, r=0.0, p=0.0, _y_=0.0):
            self.x = x  # X坐标
            self.y = y  # Y坐标
            self.z = z  # Z坐标
            self.r = r  # 旋转角度r
            self.p = p  # 旋转角度p
            self._y_ = _y_  # 旋转角度y
def normalize_arm2_position(robot_client,sock):
        print("===  机器人2控制 ===")
        ptp_target = Pose(x=135.429, y=-150.135, z=1073.908, r=-89.964, p=-2.869, _y_=-137.097)
        # 使用默认阈值（x/y/z:±1mm）
        ptp_res = robot_client.run_point_with_safety(
            sock,
            p=ptp_target,
            speed=20,
            tolerance={'x': 1.5, 'y': 1.5}  # 自定义x/y轴阈值
        )
        print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
        print(
            f"目标位置：x={ptp_res['target_pose'].x:.2f}, y={ptp_res['target_pose'].y:.2f}, z={ptp_res['target_pose'].z}")
        print(
            f"实际位置：x={ptp_res['verification']['current_pose'].x:.2f}, y={ptp_res['verification']['current_pose'].y:.2f}, z={ptp_res['verification']['current_pose'].z:.2f}")
        print(
            f"误差：x={ptp_res['verification']['errors']['x']:.2f}mm, y={ptp_res['verification']['errors']['y']:.2f}mm, z={ptp_res['verification']['errors']['z']:.2f}mm")
        print(f"耗时：{ptp_res['total_time']}秒，信息：{ptp_res['verification']['message']}")
def move_arm1_photo_pose(robot_client: RobotAPIClient) -> bool:#拍总体图时机械臂姿态
    try:
        print("===  机器人控制 ===")
        print(f"连接状态：{robot_client.get_status()}")
        print(f"当前TCP位姿：{robot_client.get_tcp_pose()}")
        # 需要改一下.........
        target_pose = {
            "x": 164.629, "y": -335.179, "z": 443.237,
            "r": -102.172, "p": -1.055, "_y_": -138.053
        }
        ptp_result = robot_client.run_point(
            pose=target_pose, speed=30, tolerance={"x": 1.5, "y": 1.5}
        )
        print(ptp_result)
        print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
        print(f"运动误差：{ptp_result['verification']['errors']}")
    except Exception as e:
        print(f"机器人控制异常：{str(e)}")
    finally:
        pass
        # print(f"断开机器人连接：{robot_client.disconnect()}")
def arm1_photo(client):#移动机械臂调用camera1拍
    status = client.get_server_status()
    if not status or not status['camera_connected']:
        print("\n相机未连接，尝试重启相机...")
        # 尝试重启相机（最多重试2次）
        restart_success = False
        for _ in range(2):
            restart_success = client.restart_camera()
            if restart_success:
                time.sleep(3)  # 等待相机重启完成
                break
            time.sleep(2)
        # 重启后再次检查状态
        if restart_success:
            status = client.get_server_status()
            if not status or not status['camera_connected']:
                print("相机重启后仍未连接，程序退出")
                raise RuntimeError("相机重启失败")
        else:
            print("相机重启失败，程序退出")
            raise RuntimeError("相机重启失败")
    # 3. 第二步：请求并获取彩色+深度图像
    print("\n=== 开始请求图像 ===")
    image_data = client.request_both_images(save_local=True)  # save_local=True 保存到本地
    if not image_data:
        print("获取图像失败，程序退出")
        raise RuntimeError("相机图像获取失败")
    # 4. 第三步：显示图像（按ESC关闭）可选 ,注释掉显示，避免阻塞
    client.show_images(
        color_img=image_data['color_img'],
        depth_img=image_data['depth_img']
    )
    # 5. 后续操作示例（如深度数据处理）
    print(f"\n=== 图像信息 ===")
    print(f"获取时间：{image_data['timestamp']}")
    print(f"图像分辨率：{image_data['resolution']}")
    print(f"彩色图形状：{image_data['color_img'].shape}，数据类型：{image_data['color_img'].dtype}")
    print(f"深度图形状：{image_data['depth_img'].shape}，数据类型：{image_data['depth_img'].dtype}")
    print("\n客户端操作完成！")
    return image_data
def get_target_row_col(box, image_width_ratio=1.0, image_height_ratio=1.0):  #根据SAM边界框判断目标所在的行和列
    """
    :param box: SAM返回的单个目标边界框字典（含xmin/ymin等）
    :param image_width_ratio: 图像宽度相对比例（默认1.0）
    :param image_height_ratio: 图像高度相对比例（默认1.0）
    :return: 目标所在行索引（1/2/3）和列索引（1/2/3）
    """
    # 列判断
    xmin = box["box"]["xmin"]
    column1_max = image_width_ratio / 3  # 第1列：0 ~ 1/3
    column2_max = image_width_ratio * 2 / 3  # 第2列：1/3 ~ 2/3
    if xmin < column1_max:
        column = 1
    elif xmin < column2_max:
        column = 2
    else:
        column = 3

    # 行判断
    ymin = box["box"]["ymin"]
    row1_max = image_height_ratio *0.30  # 第1行（顶部区域）
    row2_max = image_height_ratio * 0.55 # 第2行
    if ymin < row1_max:
        row = 1
    elif ymin < row2_max:
        row = 2
    else:
        row = 3

    return row, column
def judge_all_target_rows_cols(sam_result, image_shape):#批量判断所有识别目标的所在行和列
    """
    :param sam_result: SAM调用返回的完整结果字典
    :param image_shape: 图像形状（h, w, c）
    :return: 字典列表，含每个目标的标签、置信度、所在行、所在列
    """
    result_list = []
    if sam_result["status"] != "success" or sam_result["detected_objects"] == 0:
        print("SAM未检测到有效目标")
        return result_list

    image_height, image_width = image_shape[0], image_shape[1]
    print(f"图像尺寸：{image_width}px（宽）× {image_height}px（高）")


    for box in sam_result["detected_boxes"]:
        row, column = get_target_row_col(box)
        result = {
            "label": box["label"],
            "confidence": round(box["confidence"], 2),
            "xmin": round(box["box"]["xmin"], 3),  # 列判断依据
            "ymin": round(box["box"]["ymin"], 3),  # 行判断依据
            "row": row,          # 所在行（1/2/3）
            "column": column     # 所在列（1/2/3）
        }
        result_list.append(result)
        print(f"目标：{result['label']} | 置信度：{result['confidence']} | 位置：第{row}行，第{column}列")

    return result_list
def analyze_targets(image_data):   #分析图像中的目标并判断所在列
    print("\n=== 开始目标检测与列分析 ===")
    timestamp = image_data['timestamp'].replace(":", "").replace(" ", "_").replace(".", "_")
    # SAM结果保存目录：与原始图像同目录（output/new2/时间戳子目录）
    sam_save_dir = os.path.join("output/new2", timestamp.split("_")[0] + "_" + timestamp.split("_")[1])
    sam_result = call_grounded_sam_service(
        service_url="http://192.168.1.19:1236/process",
        image=image_data['color_img'],
        text_prompt="Red cube.     Light blue cube.   Cyan block.   Purple cube.  Pink cube. ",
        box_threshold=0.395,
        text_threshold=0.25,
        save_result_images=True,
        save_masks=True,
        display_results=False,
        save_dir=sam_save_dir
    )
    target_columns = judge_all_target_rows_cols(sam_result, image_data['color_img'].shape)
    return target_columns,sam_result
def get_unique_columns(target_columns):  #列重拍列
    """
    从分析结果中提取有物品的列（去重，确保每列只处理一次）
    :param target_columns: 目标分析结果列表
    :return: 排序后的有效列列表（如[1,3]）
    """
    # 提取所有列并去重
    columns = {target["column"] for target in target_columns}
    # 按1→2→3的顺序排序
    sorted_columns = sorted(columns)
    print(f"\n有效列（去重后）：{sorted_columns}")
    return sorted_columns
def move_chassis_to_column(chassis_client: WooshApi, column: int) -> bool:  #移动机械臂1去取物位置
    """
    控制机器人底盘移动到指定列的预设坐标
    :param chassis_client: 底盘控制客户端
    :param column: 目标列（1/2/3）
    :return: 移动是否成功
    """
    # 预设底盘在每列的坐标（x, y, theta），请替换为你的实际坐标！
    # 例如：第1列底盘位置、第2列底盘位置、第3列底盘位置
    column_chassis_positions = {
        1: (0.96, -1.25, 3.14),  # (x, y, 角度)
        2: (0.96, -2.2, 3.14),
        3: (0.96, -3.4, 3.14)
    }
    if column not in column_chassis_positions:
        print(f"错误：列{column}无预设底盘坐标")
        return False
    try:
        print(f"\n=== 底盘移动到第{column}列 ===")
        x, y, theta = column_chassis_positions[column]
        # 调用底盘移动接口（根据你的WooshApi实际方法调整）
        result = chassis_client.robot_go_to(x=x, y=y, theta=theta)
        print(f"底盘移动指令已发送：x={x}, y={y}, theta={theta}")
        # 等待移动完成（根据实际移动时间调整，或监听完成信号）
        print("底盘移动完成")
        return True
    except Exception as e:
        print(f"底盘移动异常：{str(e)}")
        return False
def move_arm1_to_target(robot_client: RobotAPIClient, row: int, column: int):   #移动机械臂1摆不同的取物姿态
    """
    控制机械臂移动到对应行列的目标位置（示例姿态，需根据实际场景调整）
    :param robot_client: 机械臂控制客户端
    :param row: 目标所在行（1/2/3）
    :param column: 目标所在列（1/2/3）
    :return: 操作是否成功
    """
    try:
        print(f"\n=== 控制机械臂移动到第{row}行第{column}列目标位置 ===")
        # 格式：x, y, z, r, p, _y_（根据实际机械臂坐标系调整）
        target_poses = {
            (1, 1): {"x":196.93, "y": -298.62, "z": 350.03, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (1, 2): {"x":196.93, "y": -298.62, "z": 350.03, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (1, 3): {"x":196.93, "y": -298.62, "z": 350.03, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (2, 1): {"x": 264.329, "y": -391.236, "z": 581.466, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (2, 2): {"x": 264.329, "y": -391.236, "z": 581.466, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (2, 3): {"x": 264.329, "y": -391.236, "z": 581.466, "r": -133.659, "p": 0.42, "_y_": -130.379},
            (3, 1): {"x": 225.16, "y": -352.58, "z":649.92, "r": -89.10, "p":0.42, "_y_": -140.07},
            (3, 2): {"x": 225.16, "y": -352.58, "z":649.92, "r": -89.10, "p":0.42, "_y_": -140.07},
            (3, 3): {"x": 225.16, "y": -352.58, "z":649.92, "r": -89.10, "p":0.42, "_y_": -140.07},
        }
        # 获取当前行列对应的目标姿态
        target_pose = target_poses.get((row, column))
        if not target_pose:
            raise Exception(f"未定义第{row}行第{column}列的目标姿态")
        # 移动机械臂到目标姿态
        ptp_result = robot_client.run_point(
            pose=target_pose,
            speed=30,
            tolerance={"x": 1.5, "y": 1.5}
        )
        print(f"机械臂移动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
        print(f"运动误差：{ptp_result['verification']['errors']}")
        return ptp_result['verification']['success']
    except Exception as e:
        print(f"机械臂姿态调整异常：{str(e)}")
        return False
    finally:
        pass
def align_images(color, depth, mask):#对齐三张图
    #用cv2
    h, w = depth.shape[:2]#depth 图决定统一尺寸，因为深度图一般由深度相机给出，是真正的几何基准
    # 对齐 color
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)#如果彩色图大小 ≠ 深度图大小，就给它缩放成相同尺寸。使用 双线性插值（INTER_LINEAR），适合 RGB 图像。
    # 对齐 mask
    if mask is not None:
        if len(mask.shape) == 3 and mask.shape[2] > 1:#如果 mask 是 3 通道（例如：H×W×3） → 只取一个通道
            #mask.shape = (h, w, c)，c是通道数
            mask = mask[:, :, 0]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)#or mask = (mask > 127).astype(np.uint8)  # 阈值从0→127
    return color, depth, mask
def calculate_foot_of_perpendicular_to_tcp_plane(tcp_pose, object_point):
    """
    计算物体点到TCP平面L的垂足C的坐标
    :param tcp_pose: TCP的位姿（含位置和姿态），格式：[x1, y1, z1, r_deg, p_deg, y_deg]
                    x1,y1,z1: TCP的3D坐标（mm）
                    r_deg,p_deg,y_deg: TCP的滚转/俯仰/偏航角 弧度
    :param object_point: 物体的3D坐标（mm），格式：[x2, y2, z2]
    :return: 垂足C的3D坐标（mm），格式：[cx, cy, cz]
    """
    # 1. 解析输入参数
    x1, y1, z1 = tcp_pose[0], tcp_pose[1], tcp_pose[2]
    r, p, y = tcp_pose[3], tcp_pose[4], tcp_pose[5]
    x2, y2, z2 = object_point[0], object_point[1], object_point[2]
    # 3. 计算TCP的旋转矩阵
    # 旋转矩阵作用：将世界坐标系的向量转换为TCP坐标系的向量
    R_yaw = np.array([[cos(y), -sin(y), 0],
                      [sin(y), cos(y), 0],
                      [0, 0, 1]])  # 偏航角旋转矩阵
    R_pitch = np.array([[cos(p), 0, sin(p)],
                        [0, 1, 0],
                        [-sin(p), 0, cos(p)]])  # 俯仰角旋转矩阵
    R_roll = np.array([[1, 0, 0],
                       [0, cos(r), -sin(r)],
                       [0, sin(r), cos(r)]])  # 滚转角旋转矩阵
    # 总旋转矩阵  执行顺序（先→后）：先 X（滚转）→ 再 Y（俯仰）→ 最后 Z（偏航）
    #R是小坐标到大坐标的旋转矩阵
    R = R_yaw @ R_pitch @R_roll
    # 1. 小坐标系的方向向量：TCP局部Z轴（垂直于TCP平面，即法向量方向）
    tcp_local_z = np.array([0, 0, 1])  # 小坐标系里，Z轴方向就是[0,0,1]
    # 2. 用旋转矩阵R转换：小→大
    n_world = R @ tcp_local_z  # 结果是大坐标系里的法向量
    # 3. 后续使用：用这个大坐标系的法向量，计算物体到TCP平面的垂足
    a, b, c = n_world  # 法向量的三个分量，用于联立平面方程
    # 5. 平面L的方程（点法式）：a*(x - x1) + b*(y - y1) + c*(z - z1) = 0
    # 6. 直线h的参数方程（过物体点，方向向量为法向量n）：
    #    x = x2 + t*a
    #    y = y2 + t*b
    #    z = z2 + t*c
    # 7. 联立平面方程和直线方程，求解参数t（垂足对应的t值）
    numerator = a*(x1 - x2) + b*(y1 - y2) + c*(z1 - z2)
    denominator = a**2 + b**2 + c**2
    t = numerator / denominator  # 关键参数：直线h上垂足对应的参数
    # 8. 计算垂足C的坐标（代入直线参数方程）
    cx = x2 + t * a
    cy = y2 + t * b
    cz = z2 + t * c
    # 8. 计算垂足C的坐标 + 安全计算斜率
    cx = x2 + t * a
    cy = y2 + t * b
    cz = z2 + t * c
    return [cx, cy, cz]
def interpolate_points(start_point, end_point, num_points=8):#机械臂插值
    """
    在两点之间生成线性插值点（支持二维或三维）
    start_point: 起点坐标 (x1, y1) 或 (x1, y1, z1)
    end_point:   终点坐标 (x2, y2) 或 (x2, y2, z2)
    num_points: 插值点数量
    返回形状为 (num_points, n) 的数组，每行是一个点
    """
    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)
    # 自动适配维度（2D 或 3D）
    dims = len(start_point)
    # 对每个维度插值
    result = np.zeros((num_points, dims))
    for i in range(dims):
        result[:, i] = np.linspace(start_point[i], end_point[i], num_points)
    return result
def arm1_get_from_shelf(camera1,arm1_pose,row,gripper):#识取并回位
    center_point_list = []
    input("\n=== 开始请求camera1图像 ===")
    image_data = camera1.request_both_images(save_local=False)  # save_local=True 保存到本地
    if not image_data:
        print("获取图像失败，程序退出")
        raise RuntimeError("获取图像失败")
    cam1_color_img = image_data['color_img']
    cam1_depth_img = image_data['depth_img']
    input('开始根据图像计算物体坐标！')
    color_img = cam1_color_img
    depth_img = cam1_depth_img
    # 配置调用物体检测的参数
    config = {
        "service_url": "http://192.168.1.19:1236/process",
        "image": color_img,
         "text_prompt": "           A PINK BLOCK.        A PURPLE BLOCK.         A RED BLOCK.      ",  # 检测
         "box_threshold": 0.38,
          "text_threshold": 0.25,
        "save_result_images": True,
        "save_masks": True,
        "display_results": False,
        "save_dir": "output/cam1_single"
    }
    # 生成点云
    # camera1内参: fx, fy, cx, cy = 604.95, 604.95, 316.23, 233.86  # camera2内参: fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67
    fx, fy, cx, cy = 604.95, 604.95, 316.23, 233.86  # 彩色内参
    tcp_pose = arm1_pose.get_tcp_pose()
    if not tcp_pose.get("success"):
        print(f"获取TCP位姿失败: {tcp_pose.get('message', '未知错误')}")
        raise RuntimeError("获取TCP位姿失败失败")
    x = tcp_pose["pose"]["x"]
    y = tcp_pose["pose"]["y"]
    z = tcp_pose["pose"]["z"]
    r_deg = tcp_pose["pose"]["r"]  # 翻滚角（度）
    p_deg = tcp_pose["pose"]["p"]  # 俯仰角（度）
    y_deg = tcp_pose["pose"]["_y_"]  # 偏航角（度）
    arm_tcp = [x, y, z, r_deg / 180 * pi, p_deg / 180 * pi, y_deg / 180 * pi]
    print("获取armtcp是:", x, y, z, r_deg, p_deg, y_deg)
    cam_tcp_pose = [30, -100, 30, 0 / 180 * pi, 0 / 180 * pi, -180 / 180 * pi]
    # 调用GroundedSAM服务
    input('调用GroundedSAM服务')
    service_result = call_grounded_sam_service(**config)
    # 解析结果（重点：使用顺序一致的boxes和masks）
    if service_result["status"] == "success":
        print(f"\n=== 客户端结果解析 ===")
        print(f"检测目标总数：{service_result['detected_objects']}")
        print(f"保存的结果图像：{service_result['saved_images']}")
        boxes = service_result["detected_boxes"]
        masks = service_result["detected_masks"]
        input("\n=== SAM 结果检查 ===")
        print("检测框数量:", len(boxes))
        print("掩码数量:", len(masks))
        center_point_list = []
        if len(boxes) > 0 and len(boxes) == len(masks):
            print(f"识别框与掩码关联")
            for idx, (box, mask) in enumerate(zip(boxes, masks)):
                print(f"目标 {idx + 1}（ID：{box['id']}）")
                print(f"标签：{box['label']} | 置信度：{box['confidence']}")
                print(
                    f"识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                print(
                    f"掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")
                # 创建点云生成器实例
                pcd_generator = PointCloudGenerator(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    visualize=True,
                    save_point_cloud=False,
                    tcp_pose=arm_tcp,  # 传入含旋转的TCP坐标
                    camera_to_tcp_pose=cam_tcp_pose,  # 传入相机相对于TCP的6D位姿
                    save_path="generated_point_cloud.npy"
                )
                color_img, depth_img, mask_img = align_images(color_img, depth_img, mask['mask_data'])
                print("\n=== 带掩码模式 ===")
                mask_response = pcd_generator.generate_point_cloud(
                    color_image_ori=color_img,
                    depth_image_ori=depth_img,
                    mask=mask_img,
                    downsample_scale=1
                )
                if mask_response["state"] == "success":
                    print(f"生成结果：{mask_response['info']}")
                    input("\n=== 掩码点云检查 ===")
                    mask_bounds = pcd_generator.get_point_cloud_bounds()
                    print("x_min/x_max:", mask_bounds['x_min'], mask_bounds['x_max'])
                    print("y_min/y_max:", mask_bounds['y_min'], mask_bounds['y_max'])
                    print("z_min/z_max:", mask_bounds['z_min'], mask_bounds['z_max'])
                    mask_center = pcd_generator.get_point_cloud_center(use_bounds=False)  # 用均值计算中心点
                    print(f"掩码点云最值：x[{mask_bounds['x_min']:.2f}, {mask_bounds['x_max']:.2f}], "
                          f"y[{mask_bounds['y_min']:.2f}, {mask_bounds['y_max']:.2f}], "
                          f"z[{mask_bounds['z_min']:.2f}, {mask_bounds['z_max']:.2f}]")
                    print(
                        f"掩码点云均值中心点：({mask_center[0]:.2f}, {mask_center[1]:.2f}, {mask_center[2]:.2f})")
                    center_point_list.append([mask_center[0], mask_center[1], mask_center[2]])
    else:
        print(f"=== 调用失败 ===")
        print(f"原因：{service_result['message']}")
    if center_point_list:
        # 取第一个检测到的物体中心点（可根据需求调整）
        obj_x, obj_y, obj_z = center_point_list[0]
        # 设置抓取偏移
        # grasp_offset = [-130, 130, 0]  # mm
        # obj_x, obj_y, obj_z = obj_x + grasp_offset[0], obj_y + grasp_offset[1], obj_z + grasp_offset[2]
        #
        if row == 3:
            obj_x,obj_y=obj_x+ (-130),obj_y+(130)
            obj_z = obj_z + 30
        elif row == 2:
            obj_x, obj_y = obj_x + (-95), obj_y + (95)
            # obj_x, obj_y = obj_x + (-100), obj_y + (90)
            obj_z = obj_z + 145
        elif row == 1:
            obj_x, obj_y = obj_x + (-100), obj_y + (90)
            obj_z = obj_z + 150  # 到时候定
        arm_xyz = [x, y, z]
        obj_xyz = [obj_x, obj_y, obj_z]
        # 计算交点
        intersection_xyz = calculate_foot_of_perpendicular_to_tcp_plane(arm_tcp, obj_xyz)
        # 生成机械臂到交点插值路径
        path_to_intersection = interpolate_points(arm_xyz, intersection_xyz, num_points=8)
        # 生成交点到物体插值路径
        path_to_object = interpolate_points(intersection_xyz, obj_xyz, num_points=8)
        # 合并两段路径，形成完整移动路径
        full_path = np.vstack((path_to_intersection, path_to_object))
        # 循环沿插值路径逐点移动
        input("动")
        for idx, point in enumerate(full_path):
            x_curr, y_curr, z_curr = point
            # 沿用当前机械臂姿态，只修改位置
            pose = {
                "x": x_curr,
                "y": y_curr,
                "z": z_curr,
                "r": r_deg,
                "p": p_deg,
                "_y_": y_deg
            }
            print(f"\n移动到路径点 {idx + 1}/{len(full_path)}: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})")
            try:
                # input("动")
                ptp_result = arm1_pose.run_point(
                    pose=pose,
                    speed=10,  # 可根据需要调整速度
                    tolerance={"x": 1.0, "y": 1.0, "z": 1.0}
                )
                if not ptp_result['verification']['success']:
                    print(f"运动误差：{ptp_result['verification']['errors']}")
            except Exception as e:
                print(f"移动机械臂时发生错误：{str(e)}")
        input("arm1合爪子")
        gripper.gripper_catch()
        input("原路返回......")
        reverse_path = full_path[::-1]
        ls=[]#记录路径，用于等会放回用
        for idx, point in enumerate(reverse_path):
            x_curr, y_curr = point[0],point[1]
            # 沿用当前机械臂姿态，只修改位置
            pose = {
                "x": x_curr,
                "y": y_curr,
                "z": obj_z+25,
                "r": r_deg,
                "p": p_deg,
                "_y_": y_deg
            }
            ls.append(pose)
            print(f"\n移动到路径点 {idx + 1}/{len(reverse_path)}: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})")
            try:
                # input("动")
                ptp_result = arm1_pose.run_point(
                    pose=pose,
                    speed=10,  # 可根据需要调整速度
                    tolerance={"x": 1.0, "y": 1.0, "z": 1.0}
                )
                if not ptp_result['verification']['success']:
                    print(f"运动误差：{ptp_result['verification']['errors']}")

            except Exception as e:
                print(f"移动机械臂时发生错误：{str(e)}")
        return ls
def arm1_put_on_car1(robot_client,gripper_client):#放下松爪到小车,并原路返回
    #通过小车自身定位和arm1底盘定位确定，映射到arm1的TCP位姿
   for i in range(2):
       if i==0:
           input("放下第1步....")
           try:
               target_pose = {
                   "x": 18.68, "y": -126.35, "z": 917.5,
                   "r": -88.1, "p": -1.0, "_y_": -42.4
               }#2
               ptp_result = robot_client.run_point(
                   pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
               )
               print(ptp_result)
               print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
               print(f"运动误差：{ptp_result['verification']['errors']}")
           except Exception as e:
               print(f"机器人控制异常：{str(e)}")
       elif i==1:
           input("放下第2步.....")
           try:
               target_pose = {
                   "x": 609.67, "y":495, "z": 170.74,
                   "r": -178.51, "p": -1.87, "_y_": -42.2
               }  # 3
               ptp_result = robot_client.run_point(
                   pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
               )
               print(ptp_result)
               print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
               print(f"运动误差：{ptp_result['verification']['errors']}")
           except Exception as e:
               print(f"机器人控制异常：{str(e)}")
   input("arm1松爪......")
   gripper_client.gripper_release()
   for i in range(2):
       if i==0:
           input("返回第1步....")
           try:
               target_pose = {
                   "x": 18.68, "y": -126.35, "z": 917.5,
                   "r": -88.1, "p": -1.0, "_y_": -42.4
               }#2
               ptp_result = robot_client.run_point(
                   pose=target_pose, speed=30, tolerance={"x": 1.5, "y": 1.5}
               )
               print(ptp_result)
               print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
               print(f"运动误差：{ptp_result['verification']['errors']}")
           except Exception as e:
               print(f"机器人控制异常：{str(e)}")
       elif i==1:
           input("返回第2步.....")
           try:
               target_pose = {
                   "x": -129.0, "y": -12.0, "z": 915.0,
                   "r": -89.0, "p": -1.0, "_y_": -135.0
               }#标准姿态
               ptp_result = robot_client.run_point(
                   pose=target_pose, speed=30, tolerance={"x": 1.5, "y": 1.5}
               )
               print(ptp_result)
               print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
               print(f"运动误差：{ptp_result['verification']['errors']}")
           except Exception as e:
               print(f"机器人控制异常：{str(e)}")
def amr1_get_from_car1(arm1,gripper1,camera1,path):
            input("下放到小车拍摄位")
            input("arm1松爪......")
            gripper1.gripper_release()
            input("放下第1步....")
            try:
                target_pose = {
                    "x": 18.68, "y": -126.35, "z": 917.5,
                    "r": -88.1, "p": -1.0, "_y_": -42.4
                }  # 1
                ptp_result = arm1.run_point(
                    pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
                print(ptp_result)
                print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
                print(f"运动误差：{ptp_result['verification']['errors']}")
            except Exception as e:
                print(f"机器人控制异常：{str(e)}")
            input("放下第2步.....")
            try:
                #1111111111111111111111111111111111111111111111111111改
                target_pose_cam1= {
                    "x": 518.032, "y":327.077, "z": 455.703,
                    "r":174.739, "p": -1.741, "_y_": -46.651
                }
                arm1.run_point(
                    pose=target_pose_cam1, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
            except Exception as e:
                print(f"机器人控制异常：{str(e)}")
            # 拍
            input("\n=== 开始请求camera1图像 ===")
            image_data = camera1.request_both_images(save_local=False)  # save_local=True 保存到本地
            if not image_data:
                print("获取图像失败，程序退出")
                raise RuntimeError("图像失败失败")
            cam1_color_img = image_data['color_img']
            cam1_depth_img = image_data['depth_img']
            input('根据图像计算物体坐标！')
            color_img = cam1_color_img
            depth_img = cam1_depth_img
            # 配置调用物体检测的参数
            config = {
                "service_url": "http://192.168.1.19:1236/process",
                "image": color_img,  # 替换为你的图像路径
                "text_prompt": "A SMALL PINK BLOCK.      A  SMALL RED BLOCK.   A  SMALL PURPLE BLOCK  ",  # 检测
                "box_threshold": 0.4,
                "text_threshold": 0.25,
                "save_result_images": True,
                "save_masks": True,
                "display_results": False,
                "save_dir": "output/new3"
            }
            fx, fy, cx, cy = 604.95, 604.95, 316.23, 233.86  # 彩色内参
            tcp_pose = arm1.get_tcp_pose()
            if not tcp_pose.get("success"):
                print(f"获取TCP位姿失败: {tcp_pose.get('message', '未知错误')}")
                raise RuntimeError("获取tcp失败")
            x = tcp_pose["pose"]["x"]
            y = tcp_pose["pose"]["y"]
            z = tcp_pose["pose"]["z"]
            r_deg = tcp_pose["pose"]["r"]  # 翻滚角（度）
            p_deg = tcp_pose["pose"]["p"]  # 俯仰角（度）
            y_deg = tcp_pose["pose"]["_y_"]  # 偏航角（度）
            arm_tcp = [x, y, z, r_deg / 180 * pi, p_deg / 180 * pi, y_deg / 180 * pi]
            print("armtcp:", x, y, z, r_deg, p_deg, y_deg)
            cam_tcp_pose = [30, -100, 30, 0 / 180 * pi, 0 / 180 * pi, -180 / 180 * pi]
            # 计算
            input("计算")
            # 调用GroundedSAM服务
            service_result = call_grounded_sam_service(**config)
            # 解析结果（重点：使用顺序一致的boxes和masks）
            if service_result["status"] == "success":
                print(f"\n=== 客户端结果解析 ===")
                print(f"检测目标总数：{service_result['detected_objects']}")
                print(f"保存的结果图像：{service_result['saved_images']}")
                # 关联使用识别框与掩码（索引一致）
                boxes = service_result["detected_boxes"]
                masks = service_result["detected_masks"]
                center_point_list = []
                if len(boxes) > 0 and len(boxes) == len(masks):
                    print(f"识别框与掩码关联")
                    for idx, (box, mask) in enumerate(zip(boxes, masks)):
                        print(f"目标 {idx + 1}（ID：{box['id']}）")
                        print(f"标签：{box['label']} | 置信度：{box['confidence']}")
                        print(
                            f"识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                        print(
                            f"掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")
                        # 创建点云生成器实例
                        pcd_generator = PointCloudGenerator(
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            visualize=False,
                            save_point_cloud=False,
                            tcp_pose=arm_tcp,  # 传入含旋转的TCP坐标
                            camera_to_tcp_pose=cam_tcp_pose,  # 传入相机相对于TCP的6D位姿
                            save_path="generated_point_cloud.npy"
                        )
                        print("\n=== 带掩码模式 ===")
                        mask_response = pcd_generator.generate_point_cloud(
                            color_image_ori=color_img,
                            depth_image_ori=depth_img,
                            mask=mask['mask_data'],
                            downsample_scale=1
                        )
                        if mask_response["state"] == "success":
                            print(f"生成结果：{mask_response['info']}")
                            # 直接调用子函数获取最新点云的最值和中心点（无需重复传参，自动用缓存）
                            mask_bounds = pcd_generator.get_point_cloud_bounds()
                            mask_center = pcd_generator.get_point_cloud_center(use_bounds=False)  # 用均值计算中心点
                            print(f"掩码点云最值：x[{mask_bounds['x_min']:.2f}, {mask_bounds['x_max']:.2f}], "
                                  f"y[{mask_bounds['y_min']:.2f}, {mask_bounds['y_max']:.2f}], "
                                  f"z[{mask_bounds['z_min']:.2f}, {mask_bounds['z_max']:.2f}]")
                            print(
                                f"掩码点云均值中心点：({mask_center[0]:.2f}, {mask_center[1]:.2f}, {mask_center[2]:.2f})")
                            center_point_list.append([mask_center[0], mask_center[1], mask_center[2]])
                else:
                    print(f"无有效识别框或掩码（数量不匹配）")
            else:
                print(f"=== 调用失败 ===")
                print(f"原因：{service_result['message']}")
            input("回车键估计物体旋转角")
            # -------------------------- 用户配置（关键！请根据实际情况修改） --------------------------
            cloud_np = np.asarray(mask_response["point_cloud"])
            cube_side = 0.1  # 立方体实际边长（米）
            real_yaw = None  # 真实Yaw角（可选，用于误差验证）
            z_tolerance = 0.01  # 上表面Z值公差（米，噪声大时调大）
            # ----------------------------------------------------------------------------------------
            # 1. 判断读取外部点云（毫米单位）
            if cloud_np.shape[1] != 3:
                raise ValueError(f"点云维度错误！需为3D点云（N×3），当前格式：{cloud_np.shape}")
            print(f"[外部点云读取完成]")
            print(f"  - 点云点数：{len(cloud_np)}")
            # 2. 初始化姿态估计器并执行估计
            estimator = CubePoseEstimator(
                cube_side=cube_side,
                save_result=True,  # 保存结果到文件
                visualize=False  # 显示可视化窗口
            )
            result = estimator.load_point_cloud(
                point_cloud_mm=cloud_np,
                real_yaw=real_yaw,
                z_tolerance=z_tolerance
            ).estimate_pose()
            cur_yaw =target_pose_cam1["_y_"]
            print("当前末端的旋转角为：", cur_yaw)
            object_yaw = result["yaw_angle"]
            print("估计出的旋转角为： ", object_yaw, "°")
            adjusted_yaw = adjust_angle_to_range(object_yaw, current_yaw=cur_yaw, lower_threshold=-45,
                                                 upper_threshold=45)
            print("最终计算出的旋转角为： ", adjusted_yaw, "°")
            # 摆
            input("回车键继续执行调整位置,并调整抓取姿态。")
            pose = {
                "x": mask_center[0],
                "y": mask_center[1],
                "z": mask_bounds['z_max'] + 300,
                "r": 180.0,
                "p": 0,
                "_y_": adjusted_yaw
            }
            # 使用默认阈值（x/y/z:±1mm）
            arm1.run_point(
                pose=pose, speed=20, tolerance={"x": 1.0, "y": 1.0, "z": 1.0}  # 自定义x/y轴阈值
            )
            # 下放夹
            input("回车键继续执行抓取动作")
            pose={
                "x": mask_center[0],
                "y": mask_center[1],
                "z": mask_bounds['z_max'] + 160,
                "r": 180.0,
                "p": 0,
                "_y_": adjusted_yaw
            }
            arm1.run_point(
                pose=pose, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
            )
            input("gripper1 执行抓取")
            gripper1.gripper_catch()
            input("回拍摄位")
            try:
                arm1.run_point(
                    pose=target_pose_cam1, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
            except Exception as e:
                print(f"机器人控制异常：{str(e)}")
            input("返回....")
            try:
                target_pose = {
                    "x": 18.68, "y": -126.35, "z": 917.5,
                    "r": -88.1, "p": -1.0, "_y_": -42.4
                }
                arm1.run_point(
                    pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
            except Exception as e:
                print(f"机器人控制异常：{str(e)}")
            input("amr1标准化")
            normalize_arm1_position(arm1)
            #回对应拍摄位
            for idx,pose in enumerate(path[::-1]):
                # input("返回.....")
                target_pose = {
                    "x": pose["x"], "y":pose["y"], "z":pose["z"]+20,
                    "r":pose["r"], "p": pose["p"], "_y_": pose["_y_"]
                }
                print(f"返回第{idx}步")
                arm1.run_point(
                    pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
            input("arm1松爪......")
            gripper1.gripper_release()
            input("返回完毕,回位......")
            for idx, pose in enumerate(path):
                print(f"返回第{idx}步")
                target_pose = {
                    "x": pose["x"], "y": pose["y"], "z": pose["z"] + 20,
                    "r": pose["r"], "p": pose["p"], "_y_": pose["_y_"]
                }
                arm1.run_point(
                    pose=target_pose, speed=20, tolerance={"x": 1.5, "y": 1.5}
                )
            #根据原路返回，放下
def normalize_single_angle(angle, lower_threshold, upper_threshold):
    """非递归函数：将单个角度归一化到指定范围内"""
    range_span = upper_threshold - lower_threshold
    # 处理超出范围的角度（基于周期性）
    if angle < lower_threshold:
        # 计算需要加多少个跨度才能进入范围
        cycles = (lower_threshold - angle + range_span - 1) // range_span  # 向上取整
        return angle + cycles * range_span
    elif angle > upper_threshold:
        # 计算需要减多少个跨度才能进入范围
        cycles = (angle - upper_threshold + range_span - 1) // range_span  # 向上取整
        return angle - cycles * range_span
    else:
        return angle  # 已在范围内
def adjust_angle_to_range(angle, current_yaw, lower_threshold=-180, upper_threshold=180):
    """
    将角度调整到指定工作范围内，并优先选择与当前角度最接近的等效角度
    （移除递归，修复栈溢出问题）
    """
    range_span = upper_threshold - lower_threshold
    # 非递归归一化当前角度（避免栈溢出）
    current_normalized = normalize_single_angle(current_yaw, lower_threshold, upper_threshold)
    # 归一化目标角度的基础值
    base_adjusted = normalize_single_angle(angle, lower_threshold, upper_threshold)
    # 生成可能的等效角度（±跨度，考虑周期性）
    candidates = [
        base_adjusted,
        base_adjusted + range_span,
        base_adjusted - range_span
    ]
    # 筛选出在指定范围内的候选角度
    valid_candidates = [
        cand for cand in candidates
        if lower_threshold <= cand <= upper_threshold
    ]
    # 选择与当前角度最近的有效角度
    if valid_candidates:
        return min(valid_candidates, key=lambda x: abs(x - current_normalized))
    else:
        return base_adjusted  # 极端情况（理论上不会触发）
def arm2_operration(arm2, sock2, camera2,gripper2):
    input("arm2姿势标准化...........")
    normalize_arm2_position(arm2,sock2)
    # arm2夹取到传送带，再从传送带上夹取至小车
    # 去
    input("arm2前去小车夹取")
    # 抓物体拍摄位姿，根据需要修改
    ptp_target = Pose(x=504.406, y=-446.703, z=364.895, r=-173.716, p=-1.914, _y_=-120.893)
    # 使用默认阈值（x/y/z:±1mm）
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    input("松爪")
    gripper2.release()
    # 拍
    input("\n=== 开始请求camera2图像 ===")
    image_data = camera2.request_both_images(save_local=False)  # save_local=True 保存到本地
    if not image_data:
        print("获取图像失败，程序退出")
        raise RuntimeError("相机连接失败")
    cam2_color_img = image_data['color_img']
    cam2_depth_img = image_data['depth_img']
    input('根据图像计算物体坐标！')
    color_img = cam2_color_img
    depth_img = cam2_depth_img
    # 配置调用物体检测的参数
    config = {
        "service_url": "http://192.168.1.19:1236/process",
        "image": color_img,  # 替换为你的图像路径
        "text_prompt": "A TINY  PINK BLOCK.            A TINY PURPLE BLOCK.    ",  # 检测
        "box_threshold": 0.415,
        "text_threshold": 0.25,
        "save_result_images": True,
        "save_masks": True,
        "display_results": False,
        "save_dir": "output/newcam2"
    }
    # 生成点云
    # camera2内参: fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67
    fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67
    curr_pose = Pose()
    res = arm2.get_tcp_pose(sock2, curr_pose)
    if res == -1:
        print("获取位姿失败")
    else:
        x = curr_pose.x
        y = curr_pose.y
        z = curr_pose.z
        r_deg = curr_pose.r
        p_deg = curr_pose.p
        y_deg = curr_pose._y_
        arm_tcp = [x, y, z, r_deg / 180 * pi, p_deg / 180 * pi, y_deg / 180 * pi]
        print("获取armtcp是:", x, y, z, r_deg, p_deg, y_deg)
    cam_tcp_pose = [30, -100, 30, 0 / 180 * pi, 0 / 180 * pi, -180 / 180 * pi]  # 根据硬件结构，粗略测量得到。
    # 计算
    input("计算")
    # 调用GroundedSAM服务
    service_result = call_grounded_sam_service(**config)
    # 解析结果（重点：使用顺序一致的boxes和masks）
    if service_result["status"] == "success":
        print(f"\n=== 客户端结果解析 ===")
        print(f"检测目标总数：{service_result['detected_objects']}")
        print(f"保存的结果图像：{service_result['saved_images']}")
        # 关联使用识别框与掩码（索引一致）
        boxes = service_result["detected_boxes"]
        masks = service_result["detected_masks"]
        center_point_list = []
        if len(boxes) > 0 and len(boxes) == len(masks):
            print(f"识别框与掩码关联")
            for idx, (box, mask) in enumerate(zip(boxes, masks)):
                print(f"目标 {idx + 1}（ID：{box['id']}）")
                print(f"标签：{box['label']} | 置信度：{box['confidence']}")
                print(
                    f"识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                print(
                    f"掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")
                # 创建点云生成器实例
                pcd_generator = PointCloudGenerator(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    visualize=False,
                    save_point_cloud=False,
                    tcp_pose=arm_tcp,  # 传入含旋转的TCP坐标
                    camera_to_tcp_pose=cam_tcp_pose,  # 传入相机相对于TCP的6D位姿
                    save_path="generated_point_cloud.npy"
                )
                print("\n=== 带掩码模式 ===")
                mask_response = pcd_generator.generate_point_cloud(
                    color_image_ori=color_img,
                    depth_image_ori=depth_img,
                    mask=mask['mask_data'],
                    downsample_scale=1
                )
                if mask_response["state"] == "success":
                    print(f"生成结果：{mask_response['info']}")
                    # 直接调用子函数获取最新点云的最值和中心点（无需重复传参，自动用缓存）
                    mask_bounds = pcd_generator.get_point_cloud_bounds()
                    mask_center = pcd_generator.get_point_cloud_center(use_bounds=False)  # 用均值计算中心点
                    print(f"掩码点云最值：x[{mask_bounds['x_min']:.2f}, {mask_bounds['x_max']:.2f}], "
                          f"y[{mask_bounds['y_min']:.2f}, {mask_bounds['y_max']:.2f}], "
                          f"z[{mask_bounds['z_min']:.2f}, {mask_bounds['z_max']:.2f}]")
                    print(
                        f"掩码点云均值中心点：({mask_center[0]:.2f}, {mask_center[1]:.2f}, {mask_center[2]:.2f})")
                    center_point_list.append([mask_center[0], mask_center[1], mask_center[2]])
        else:
            print(f"无有效识别框或掩码（数量不匹配）")
    else:
        print(f"=== 调用失败 ===")
        print(f"原因：{service_result['message']}")
    input("回车键估计物体旋转角")
    # -------------------------- 用户配置（关键！请根据实际情况修改） --------------------------
    cloud_np = np.asarray(mask_response["point_cloud"])
    cube_side = 0.1  # 立方体实际边长（米）
    real_yaw = None  # 真实Yaw角（可选，用于误差验证）
    z_tolerance = 0.01  # 上表面Z值公差（米，噪声大时调大）
    # ----------------------------------------------------------------------------------------
    # 1. 判断读取外部点云（毫米单位）
    if cloud_np.shape[1] != 3:
        raise ValueError(f"点云维度错误！需为3D点云（N×3），当前格式：{cloud_np.shape}")
    print(f"[外部点云读取完成]")
    print(f"  - 点云点数：{len(cloud_np)}")
    # 2. 初始化姿态估计器并执行估计
    estimator = CubePoseEstimator(
        cube_side=cube_side,
        save_result=True,  # 保存结果到文件
        visualize=False  # 显示可视化窗口
    )
    result = estimator.load_point_cloud(
        point_cloud_mm=cloud_np,
        real_yaw=real_yaw,
        z_tolerance=z_tolerance
    ).estimate_pose()
    cur_yaw = ptp_res['target_pose']._y_
    print("当前末端的旋转角为：", cur_yaw)
    object_yaw = result["yaw_angle"]
    print("估计出的旋转角为： ", object_yaw, "°")
    adjusted_yaw = adjust_angle_to_range(object_yaw, current_yaw=cur_yaw, lower_threshold=-45,
                                         upper_threshold=45)
    print("最终计算出的旋转角为： ", adjusted_yaw, "°")
    # 摆
    input("回车键继续执行调整位置,并调整抓取姿态。")
    ptp_target = Pose(x=mask_center[0], y=mask_center[1], z=mask_bounds['z_max'] + 300, r=180.0, p=0,
                          _y_=adjusted_yaw)
    # 使用默认阈值（x/y/z:±1mm）
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    # 下放夹
    input("回车键继续执行抓取动作")
    # 定义目标位姿（示例值，需根据实际场景修改）
    ptp_target_cam = Pose(x=mask_center[0], y=mask_center[1], z=mask_bounds['z_max'] + 150, r=180.0, p=0,
                      _y_=adjusted_yaw)
    # 使用默认阈值（x/y/z:±1mm）
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target_cam, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    input("gripper2 执行抓取")
    if not gripper2.catch():
        print("gripper2抓取动作失败")
    # 回拍摄位
    input("回拍摄位")
    # 定义目标位姿（示例值，需根据实际场景修改）
    ptp_target = Pose(x=504.406, y=-446.703, z=364.895, r=-173.716, p=-1.914, _y_=-120.893)
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    input("arm2标准化")
    normalize_arm2_position(arm2, sock2)
    input("arm2放到传送带")
    ptp_target = Pose(x=406.249, y=287.305, z=259.088 + 50, r=-177.259, p=-0.586, _y_=-40.973)
    arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    input("松爪")
    gripper2.release()
    input("arm2标准化")
    normalize_arm2_position(arm2, sock2)
    input("去传送带上拍摄")
    ptp_target = Pose(x=315.109, y=306.038, z=607.385, r=180.0, p=0, _y_=-50)
    arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    # 拍
    input("\n=== 开始请求camera2图像 ===")
    image_data = camera2.request_both_images(save_local=False)  # save_local=True 保存到本地
    if not image_data:
        print("获取图像失败，程序退出")
        raise RuntimeError("相机连接失败")
    cam2_color_img = image_data['color_img']
    cam2_depth_img = image_data['depth_img']
    input('根据图像计算物体坐标！')
    color_img = cam2_color_img
    depth_img = cam2_depth_img
    # 配置调用物体检测的参数
    config = {
        "service_url": "http://192.168.1.19:1236/process",
        "image": color_img,  # 替换为你的图像路径
        "text_prompt": "A  PINK BLOCK.    A PURPLE BLOCK.    A RED BLOCK.     ",  # 检测
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "save_result_images": True,
        "save_masks": True,
        "display_results": False,
        "save_dir": "output/newcam2"
    }
    # 生成点云
    # camera2内参: fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67
    fx, fy, cx, cy = 615.64, 615.94, 334.76, 243.67
    curr_pose = Pose()
    res = arm2.get_tcp_pose(sock2, curr_pose)
    if res == -1:
        print("获取位姿失败")
    else:
        x = curr_pose.x
        y = curr_pose.y
        z = curr_pose.z
        r_deg = curr_pose.r
        p_deg = curr_pose.p
        y_deg = curr_pose._y_
        arm_tcp = [x, y, z, r_deg / 180 * pi, p_deg / 180 * pi, y_deg / 180 * pi]
        print("获取armtcp是:", x, y, z, r_deg, p_deg, y_deg)
    cam_tcp_pose = [30, -100, 30, 0 / 180 * pi, 0 / 180 * pi, -180 / 180 * pi]  # 根据硬件结构，粗略测量得到。
    input("计算")
    # 调用GroundedSAM服务
    service_result = call_grounded_sam_service(**config)
    # 解析结果（重点：使用顺序一致的boxes和masks）
    if service_result["status"] == "success":
        print(f"\n=== 客户端结果解析 ===")
        print(f"检测目标总数：{service_result['detected_objects']}")
        print(f"保存的结果图像：{service_result['saved_images']}")
        # 关联使用识别框与掩码（索引一致）
        boxes = service_result["detected_boxes"]
        masks = service_result["detected_masks"]
        center_point_list = []
        if len(boxes) > 0 and len(boxes) == len(masks):
            print(f"识别框与掩码关联")
            for idx, (box, mask) in enumerate(zip(boxes, masks)):
                print(f"目标 {idx + 1}（ID：{box['id']}）")
                print(f"标签：{box['label']} | 置信度：{box['confidence']}")
                print(
                    f"识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                print(
                    f"掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")
                # 创建点云生成器实例
                pcd_generator = PointCloudGenerator(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    visualize=False,
                    save_point_cloud=False,
                    tcp_pose=arm_tcp,  # 传入含旋转的TCP坐标
                    camera_to_tcp_pose=cam_tcp_pose,  # 传入相机相对于TCP的6D位姿
                    save_path="generated_point_cloud.npy"
                )
                print("\n=== 带掩码模式 ===")
                mask_response = pcd_generator.generate_point_cloud(
                    color_image_ori=color_img,
                    depth_image_ori=depth_img,
                    mask=mask['mask_data'],
                    downsample_scale=1
                )
                if mask_response["state"] == "success":
                    print(f"生成结果：{mask_response['info']}")
                    # 直接调用子函数获取最新点云的最值和中心点（无需重复传参，自动用缓存）
                    mask_bounds = pcd_generator.get_point_cloud_bounds()
                    mask_center = pcd_generator.get_point_cloud_center(use_bounds=False)  # 用均值计算中心点
                    print(f"掩码点云最值：x[{mask_bounds['x_min']:.2f}, {mask_bounds['x_max']:.2f}], "
                          f"y[{mask_bounds['y_min']:.2f}, {mask_bounds['y_max']:.2f}], "
                          f"z[{mask_bounds['z_min']:.2f}, {mask_bounds['z_max']:.2f}]")
                    print(
                        f"掩码点云均值中心点：({mask_center[0]:.2f}, {mask_center[1]:.2f}, {mask_center[2]:.2f})")
                    center_point_list.append([mask_center[0], mask_center[1], mask_center[2]])
        else:
            print(f"无有效识别框或掩码（数量不匹配）")
    else:
        print(f"=== 调用失败 ===")
        print(f"原因：{service_result['message']}")
    input("回车键估计物体旋转角")
    # -------------------------- 用户配置（关键！请根据实际情况修改） --------------------------
    cloud_np = np.asarray(mask_response["point_cloud"])
    cube_side = 0.1  # 立方体实际边长（米）
    real_yaw = None  # 真实Yaw角（可选，用于误差验证）
    z_tolerance = 0.01  # 上表面Z值公差（米，噪声大时调大）
    # ----------------------------------------------------------------------------------------
    # 1. 判断读取外部点云（毫米单位）
    if cloud_np.shape[1] != 3:
        raise ValueError(f"点云维度错误！需为3D点云（N×3），当前格式：{cloud_np.shape}")
    print(f"[外部点云读取完成]")
    print(f"  - 点云点数：{len(cloud_np)}")
    # 2. 初始化姿态估计器并执行估计
    estimator = CubePoseEstimator(
        cube_side=cube_side,
        save_result=True,  # 保存结果到文件
        visualize=False  # 显示可视化窗口
    )
    result = estimator.load_point_cloud(
        point_cloud_mm=cloud_np,
        real_yaw=real_yaw,
        z_tolerance=z_tolerance
    ).estimate_pose()
    cur_yaw = ptp_res['target_pose']._y_
    print("当前末端的旋转角为：", cur_yaw)
    object_yaw = result["yaw_angle"]
    print("估计出的旋转角为： ", object_yaw, "°")
    adjusted_yaw = adjust_angle_to_range(object_yaw, current_yaw=cur_yaw, lower_threshold=-45,
                                         upper_threshold=45)
    print("最终计算出的旋转角为： ", adjusted_yaw, "°")
    # 抓
    input("回车键继续执行调整位置，并调整抓取姿态。")
    # 定义目标位姿（示例值，需根据实际场景修改）
    ptp_target = Pose(x=mask_center[0], y=mask_center[1], z=mask_bounds['z_max'] + 300, r=180.0, p=0,
                      _y_=adjusted_yaw)
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    # 夹
    input("回车键继续执行抓取动作")
    # 定义目标位姿（示例值，需根据实际场景修改）
    ptp_target = Pose(x=mask_center[0], y=mask_center[1], z=mask_bounds['z_max'] + 150, r=180.0, p=0,
                      _y_=adjusted_yaw)
    # 使用默认阈值（x/y/z:±1mm）
    ptp_res = arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    print(f"[gripper2] 执行抓取")
    if not gripper2.catch():
        print(f"[gripper2] 抓取动作失败")
    input("arm2标准化")
    normalize_arm2_position(arm2, sock2)
    input("放到小车上")
    input("1步")
    ptp_target = Pose(x=504.406, y=-446.703, z=364.895, r=-173.716, p=-1.914, _y_=-120.893)
    # 使用默认阈值（x/y/z:±1mm）
    arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    input("2步")
    arm2.run_point_with_safety(
        sock2, p=ptp_target_cam, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    input("松爪")
    gripper2.release()
    input("返回第一步")
    ptp_target = Pose(x=504.406, y=-446.703, z=364.895, r=-173.716, p=-1.914, _y_=-120.893)
    # 使用默认阈值（x/y/z:±1mm）
    arm2.run_point_with_safety(
        sock2, p=ptp_target, speed=20, tolerance={'x': 1, 'y': 1, 'z': 1}  # 自定义x/y轴阈值
    )
    input("arm2标准化")
    normalize_arm2_position(arm2, sock2)
def arm1_analyze_targets_on_shelf(x1, y1, x3, y3,theta, d=450,  beta_deg=45):
    #计算某物体世界坐标
    # 输入参数
    # print("请输入以下参数：")
    # x3 = float(input("目标物体-机械臂坐标系x坐标："))
    # y3 = float(input("目标物体-机械臂坐标系y坐标："))
    # x1 = float(input("底盘-世界坐标系x坐标："))
    # y1 = float(input("底盘-世界坐标系y坐标："))
    # d = float(input("底盘到机械臂基底的距离d："))
    # theta_deg = float(input("底盘旋转角度theta（度）："))
    # 返回：某物体的世界坐标
    # 你的设备固定参数：机械臂基底相对于机身的初始偏角β
    # 角度转弧度
    theta_rad = math.radians(theta)
    beta_rad = math.radians(beta_deg)
    alpha_rad = theta_rad + beta_rad  # 基底坐标系相对于世界的实际旋转角
    x1=1000*x1
    y1=1000*y1#先转化为mm
    # 步骤1：机械臂基底的世界坐标
    x2 = x1 + d * math.cos(theta_rad)
    y2 = y1 + d * math.sin(theta_rad)
    # 步骤2：目标物体的旋转分量（用实际旋转角α）
    x3_rot = x3 * math.cos(alpha_rad) - y3 * math.sin(alpha_rad)
    y3_rot = x3 * math.sin(alpha_rad) + y3 * math.cos(alpha_rad)
    # 步骤3：目标物体的世界坐标
    x4 = x2 + x3_rot
    y4 = y2 + y3_rot
    X2=x2/1000
    Y2=y2/1000
    X4=x4/1000
    Y4=y4/1000
    # 输出结果
    print("\n转换结果：")
    print(f"机械臂基底在世界坐标系的坐标：({X2:.2f}m, {Y2:.2f}m)")
    print(f"目标物体在世界坐标系的坐标：({X4:.2f}m, {Y4:.2f}m)")
    return (X4, Y4)
def calculate_single_object_coordinate(camera_client, arm_client):
    """
    识别货架上单个物体并计算其相对于机械臂基底的坐标
    :param camera_client: 相机客户端实例
    :param arm_client: 机械臂客户端实例
    :return: 物体相对于机械臂基底的坐标 (x, y, z)，失败则返回None
    """
    try:
        # 相机内参（根据实际相机调整）
        fx, fy, cx, cy = 604.95, 604.95, 316.23, 233.86  # 示例内参
        # 相机相对于TCP的位姿（根据实际安装位置调整）
        cam_tcp_pose = [30, -100, 30, 0, 0, -pi]  # [x, y, z, r, p, y]（单位：mm, 弧度）
        tcp_pose = arm_client.get_tcp_pose()
        if not tcp_pose.get("success"):
            print(f"获取TCP位姿失败: {tcp_pose.get('message', '未知错误')}")
            return None
        x_tcp = tcp_pose["pose"]["x"]
        y_tcp = tcp_pose["pose"]["y"]
        z_tcp = tcp_pose["pose"]["z"]
        r_deg = tcp_pose["pose"]["r"]  # 翻滚角（度）
        p_deg = tcp_pose["pose"]["p"]  # 俯仰角（度）
        y_deg = tcp_pose["pose"]["_y_"]  # 偏航角（度）
        # 转换角度为弧度
        arm_tcp = [x_tcp, y_tcp, z_tcp,
                   r_deg / 180 * pi,
                   p_deg / 180 * pi,
                   y_deg / 180 * pi]
        print(f"机械臂TCP位姿: x={x_tcp:.2f}, y={y_tcp:.2f}, z={z_tcp:.2f}, "
              f"r={r_deg:.2f}, p={p_deg:.2f}, y={y_deg:.2f}")

        # 2. 获取相机图像（彩色图+深度图）
        print("\n=== 获取相机图像 ===")
        image_data = camera_client.request_both_images(save_local=True)
        if not image_data:
            print("获取图像失败")
            return None
        color_img = image_data['color_img']
        depth_img = image_data['depth_img']
        print(f"获取图像成功 - 彩色图尺寸: {color_img.shape}, 深度图尺寸: {depth_img.shape}")




        # 3. 调用GroundedSAM服务检测物体
        print("\n=== 调用GroundedSAM服务 ===")
        timestamp = image_data['timestamp'].replace(":", "").replace(" ", "_").replace(".", "_")
        sam_save_dir = os.path.join("output/arm1_photo_all", timestamp)
        os.makedirs(sam_save_dir, exist_ok=True)
        config = {
            "service_url": "http://192.168.1.19:1236/process",
            "image": color_img,
            "text_prompt": "A PINK BLOCK.             A PURPLE BLOCK.    A BLUE  BLOCK ",  # 检测
            "box_threshold": 0.36,
            "text_threshold": 0.25,
            "save_result_images": True,
            "save_masks": True,
            "display_results": False,
            "save_dir": "output/cam1_single"
        }

        service_result = call_grounded_sam_service(**config)
        # 解析结果（重点：使用顺序一致的boxes和masks）
        if service_result["status"] == "success":
            print(f"\n=== 客户端结果解析 ===")
            print(f"检测目标总数：{service_result['detected_objects']}")
            print(f"保存的结果图像：{service_result['saved_images']}")
            boxes = service_result["detected_boxes"]
            masks = service_result["detected_masks"]
            input("\n=== SAM 结果检查 ===")
            print("检测框数量:", len(boxes))
            print("掩码数量:", len(masks))
            center_point_list = []
            if len(boxes) > 0 and len(boxes) == len(masks):
                print(f"识别框与掩码关联")
                for idx, (box, mask) in enumerate(zip(boxes, masks)):
                    print(f"目标 {idx + 1}（ID：{box['id']}）")
                    print(f"标签：{box['label']} | 置信度：{box['confidence']}")
                    print(
                        f"识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                    print(
                        f"掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")
                    # 创建点云生成器实例
                    pcd_generator = PointCloudGenerator(
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        visualize=True,
                        save_point_cloud=False,
                        tcp_pose=arm_tcp,  # 传入含旋转的TCP坐标
                        camera_to_tcp_pose=cam_tcp_pose,  # 传入相机相对于TCP的6D位姿
                        save_path="generated_point_cloud.npy"
                    )
                    color_img, depth_img, mask_img = align_images(color_img, depth_img, mask['mask_data'])
                    print("\n=== 带掩码模式 ===")
                    mask_response = pcd_generator.generate_point_cloud(
                        color_image_ori=color_img,
                        depth_image_ori=depth_img,
                        mask=mask_img,
                        downsample_scale=1
                    )
                    if mask_response["state"] == "success":
                        print(f"生成结果：{mask_response['info']}")
                        input("\n=== 掩码点云检查 ===")
                        mask_bounds = pcd_generator.get_point_cloud_bounds()
                        print("x_min/x_max:", mask_bounds['x_min'], mask_bounds['x_max'])
                        print("y_min/y_max:", mask_bounds['y_min'], mask_bounds['y_max'])
                        print("z_min/z_max:", mask_bounds['z_min'], mask_bounds['z_max'])
                        mask_center = pcd_generator.get_point_cloud_center(use_bounds=False)  # 用均值计算中心点
                        print(f"掩码点云最值：x[{mask_bounds['x_min']:.2f}, {mask_bounds['x_max']:.2f}], "
                              f"y[{mask_bounds['y_min']:.2f}, {mask_bounds['y_max']:.2f}], "
                              f"z[{mask_bounds['z_min']:.2f}, {mask_bounds['z_max']:.2f}]")
                        print(
                            f"掩码点云均值中心点：({mask_center[0]:.2f}, {mask_center[1]:.2f}, {mask_center[2]:.2f})")
                        center_point_list.append([mask_center[0], mask_center[1], mask_center[2]])
                # center_point_list ：若有多个，用这个

                return  center_point_list
        else:
            print(f"=== 调用失败 ===")
            print(f"原因：{service_result['message']}")
    except Exception as e:
        print(f"计算物体坐标时发生错误: {str(e)}")
        return None




def flow():
    # flow3最新是粗略的总分析在货架的哪，但是都是粗糙分析+人为预设。main2是世界坐标灵活自主分析，但是目前只能一个（还未做多物体分别识别）
    # 小车直接一个点得了
    # 具体拍摄位也用探测来的坐标好了
    # 去掉点input，全部自动化




    arm1_chassis = None  # 底盘WebSocket客户端（全程复用）
    arm1_pose = None  # 机械臂HTTP客户端（全程复用）
    arm2=None
    camera1= None  # 相机客户端（全程复用）
    camera2=None
    arm1_connected = False  # 机械臂1连接状态标记（避免重复连接/释放）
    gripper1=None
    gripper2=None
    sock2=None
    car1=None
    #写死点位要全动态识别算
    try:
        # 2. 初始化所有客户端（只执行1次）
        print("=== 初始化所有客户端 ===")
        # 底盘初始化（WebSocket，后续移动列时复用）
        arm1_chassis = WooshApi("ws://192.168.1.226:5480/", debug=False)
        # 机械臂初始化（HTTP，后续所有姿态调整复用）
        arm1_pose = RobotAPIClient("http://192.168.1.226:11223")
        # 相机初始化（后续拍摄复用）
        camera1 = D435CameraClient(
            server_ip="192.168.1.226",
            server_port=11228,
            save_dir=f'output/new'
        )
        gripper1 = GripperAPIClient("192.168.1.226",11223 )
        car1= WooshApi("ws://192.168.1.229:5480/", debug=False)
        # 3. 机械臂统一连接（只连接1次，后续全程复用，不中途断开）
        connect_result = arm1_pose.connect("169.254.128.88", 8080)
        arm2=RobotController()
        sock2 = arm2.connect_socket(ip='192.168.1.210', port=8080)
        if not sock2:
            print("arm2连接失败")
            raise RuntimeError("连接失败")
        camera2= D435CameraClient(server_ip="192.168.1.231", server_port=11225, save_dir='output/newcam2')
        gripper2 = GripperClient(
            server_url="http://192.168.1.231:11224/api",
            timeout=8,  # 局域网请求超时可适当延长
            retry_count=3  # 网络不稳定时可增加重试次数
        )
        if not connect_result.get("success"):
            raise Exception("机械臂连接失败，程序退出")
        arm1_connected = True
        print(" 机械臂连接成功（全程复用）")
        # 步骤1：机械臂标准化姿态（复用连接）
        input("arm1姿势标准化...........")
        normalize_arm1_position(arm1_pose)
        input("arm2姿势标准化...........")
        normalize_arm2_position(arm2,sock2)
        input("gripper1初始化")
        if not gripper1.init_gripper():
            raise Exception("夹爪初始化失败，跳过后续夹爪逻辑")
        input("gripper2初始化")
        if not gripper2.initialize():
            print(f"gripper2 初始化失败，终止周期")
        # 步骤2：底盘移动到拍摄点（复用底盘连接）
        input("请按回车键启动移动arm1到拍摄点...........")
        print("开始移动到点位.....")
        arm1_chassis.robot_go_to(x=2.77, y=-2.6, theta=3.14)

        # input("小车前往等待位")
        # car1.robot_go_to(x=4.0, y=-3.0, theta=3.14)
        # 步骤3：机械臂调整到拍摄姿态（复用连接）
        input("arm1拍摄姿态...........")
        move_arm1_photo_pose(arm1_pose)  # 函数内已注释掉disconnect，复用连接





        input("arm1拍摄...........")
        center_point_list= calculate_single_object_coordinate(camera1, arm1_pose)
        for i in range(len(center_point_list)):
            print(f"第{i+1}个目标物体坐标为：", center_point_list[i])
            target_point_underamr1 = center_point_list[i]  # 目前只取第一个物体，后续可扩展多个物体分别处理
            x1 = 2.77
            y1 = -2.6
            theta = (3.14 / pi) * 180
            x3 = target_point_underamr1[0]
            y3 = target_point_underamr1[1]
            z3 = target_point_underamr1[2]
            target_point_underword = arm1_analyze_targets_on_shelf(x1, y1, x3, y3, theta)
            X4 = target_point_underword[0]
            Y4 = target_point_underword[1]
            print(f"目标物体世界坐标为：X={X4}m, Y={Y4}m, Z={z3 / 1000}m")
            input("arm1姿势标准化...........")
            normalize_arm1_position(arm1_pose)
            input("arm1松爪")
            gripper1.gripper_release()
            input("移动")
            arm1_chassis.robot_go_to(X4 + 1.25, Y4 - 0.1, theta=3.14)  # +1.5为了安全距离,
            input("摆动到具体位置")
            if 500 <= z3:  # 3行
                row=3
                target_poses = {"x": 225.16, "y": -352.58, "z": 649.92, "r": -89.10, "p": 0.42, "_y_": -140.07}
                ptp_result = arm1_pose.run_point(
                pose=target_poses,
                speed=20,
                tolerance={"x": 1.5, "y": 1.5}
                             )
                print(f"机械臂移动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
                print(f"运动误差：{ptp_result['verification']['errors']}")
            elif 100 <= z3 < 500:  # 2行
                row=2
                target_poses = {"x": 264.329, "y": -391.236, "z": 581.466, "r": -133.659, "p": 1.36, "_y_": -130.379}
                ptp_result = arm1_pose.run_point(
                    pose=target_poses,
                    speed=20,
                    tolerance={"x": 1.5, "y": 1.5}
                             )
                print(f"机械臂移动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
                print(f"运动误差：{ptp_result['verification']['errors']}")
            elif 0 <= z3 < 110:  # 1行
                row=1
                target_poses = {"x": 196.93, "y": -298.62, "z": 350.03, "r": -133.659, "p": 1.36, "_y_": -130.379}
                ptp_result = arm1_pose.run_point(
                     pose=target_poses,
                speed=20,
                tolerance={"x": 1.5, "y": 1.5}
                     )
                print(f"机械臂移动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
                print(f"运动误差：{ptp_result['verification']['errors']}")
            else:
                print("测量有误，z小于0")
                raise Exception("测量有误，z小于0")

            # input("识别坐标，前往架子上物体")
            # # path获得从架子上夹好后的返回路程
            # path = arm1_get_from_shelf(camera1, arm1_pose, row, gripper1)  # 识别计算，移动抓取，合爪子，原路返回
            # # arm1标准化
            # input("arm1姿势标准化...........")
            # normalize_arm1_position(arm1_pose)
            # input("小车前往..........")
            # p = [{"x": 4.0, "y": - 3.0, "theta": 3.14}, {"x": 4.0, "y": -4.2, "theta": 3.14},
            #      {"x": 0.96, "y": - 4.2, "theta": 3.14}, {"x":  X4 + 1.25 -0.43, "y": Y4 - 0.1-0.63, "theta": 3.14}]#0.67,0.82还需再准确测测
            # car1.robot_go_to_poses(p)




            # # 放下松爪到小车,并原路返回
            # arm1_put_on_car1(arm1_pose, gripper1)
            # # 小车前往arm2
            # input("小车前往arm2")
            # p2 = [{"x": 0.96, "y": - 4.2, "theta": 3.14}, {"x": 4.0, "y": -4.2, "theta": 0},
            #       {"x": 6.45, "y": -2.4, "theta": 3.14}]
            # car1.robot_go_to_poses(p2)
            # # arm2夹取到传送带，再从传送带上夹取至小车
            # arm2_operration(arm2, sock2, camera2, gripper2)
            # # 小车返回arm1旁
            # input("小车返回arm1")
            # p3 = p2[::-1]
            # p3.append(p[-1])
            # car1.robot_go_to_poses(p3)
            # # arm1分析坐标并夹取，返回原货架
            # amr1_get_from_car1(arm1_pose, gripper1, camera1, path)
            # 小车回位
            # input("小车返回等待位")
            # car1.robot_go_to_poses(p[::-1])
            input("arm1姿势标准化...........")
            normalize_arm1_position(arm1_pose)

        print("\n=== 所有业务流程执行完成 ===")
        input("arm1回去")
        arm1_chassis.robot_go_to(x=2.77, y=-2.6, theta=3.14)





















    except Exception as e:
        # 业务流程异常时，打印日志（不影响最终资源释放）
        print(f"\n 流程执行异常：{str(e)}", file=sys.stderr)
    finally:
        # 4. 核心：所有任务完成后，一次性释放所有资源（只执行1次）
        print("\n=== 开始释放所有资源（全程只释放1次） ===")
        # 释放机械臂连接（若已连接）
        if arm1_pose and arm1_connected:
            try:
                arm1_pose.disconnect()
                print(" 机械臂连接已断开")
            except Exception as e:
                print(f" 机械臂断开失败：{str(e)}")
        # 释放底盘连接（WebSocket）
        if arm1_chassis:
            try:
                if hasattr(arm1_chassis, "disconnect"):
                    arm1_chassis.disconnect()
                elif hasattr(arm1_chassis, "close"):
                    arm1_chassis.close()
                print(" 底盘连接已断开")
            except Exception as e:
                print(f"底盘断开失败：{str(e)}")
        if gripper1:
            gripper1.deinit_gripper()
            print("gripper1资源释放")
        if car1:
            car1.close()
            print("小车1资源释放")
        if gripper2 :
            gripper2.close()
            print("gripper2资源释放")
        if sock2:
            arm2.close_socket()
            print("Socket2 closed")
if __name__ == "__main__":
    flow()

