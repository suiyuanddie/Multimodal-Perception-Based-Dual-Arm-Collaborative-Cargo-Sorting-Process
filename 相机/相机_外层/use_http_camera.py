import requests
import base64
import cv2
import numpy as np
import os
from datetime import datetime
import time


class D435CameraClient:
    def __init__(self, server_ip="192.168.1.231", server_port=11225, save_dir=f'output'):
        """
        初始化客户端
        :param server_ip: 服务端IP（从服务端日志获取，如 http://192.168.1.231:11225）
        :param server_port: 服务端端口（默认11225，与服务端一致）
        一号摄像头：server_ip="192.168.1.231", server_port=11225
        """
        self.base_url = f"http://{server_ip}:{server_port}"
        # 客户端本地存储目录（按时间创建，避免覆盖）
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"客户端图像将保存至：{self.save_dir}")

    def get_server_status(self):
        """
        获取服务端状态（相机连接、帧更新等信息）
        :return: 状态字典（含服务运行状态、相机状态、最后帧时间戳）
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)


            if response.status_code == 200:
                status = response.json()

                print("\n=== 服务端状态 ===")
                print(f"服务是否运行：{'是' if status['service_running'] else '否'}")
                print(f"相机是否连接：{'是' if status['camera_connected'] else '否'}")
                print(f"相机是否初始化：{'是' if status['camera_initialized'] else '否'}")
                print(f"临时彩色图存在：{'是' if status['output_files_status']['color_exists'] else '否'}")
                print(f"临时深度图存在：{'是' if status['output_files_status']['depth_exists'] else '否'}")
                print(f"最后帧时间戳：{status['last_frame_timestamp'] if status['last_frame_timestamp'] else '无'}")
                return status
            else:
                print(f"获取状态失败，HTTP状态码：{response.status_code}")
                return None
        except Exception as e:
            print(f"连接服务端失败：{str(e)}（请检查服务端是否启动或网络是否通畅）")
            return None
    def restart_camera(self):
        """
        发送相机重启指令（故障恢复用）
        :return: 重启结果（True/False）
        """
        try:
            response = requests.get(f"{self.base_url}/restart", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"\n相机重启结果：{'成功' if result['success'] else '失败'}")
                print(f"提示：{result['message']}")
                return result['success']
            else:
                print(f"重启请求失败，HTTP状态码：{response.status_code}")
                return False
        except Exception as e:
            print(f"发送重启指令失败：{str(e)}")
            return False


    def decode_base64_to_image(self, base64_str, is_depth=False):
        """
        将Base64编码字符串解码为图像数组
        :param base64_str: Base64编码的图像数据
        :param is_depth: 是否为深度图像（True=16位深度图，False=8位彩色图）
        :return: 图像numpy数组（彩色图：(H,W,3)，深度图：(H,W)）
        """
        try:
            # Base64解码为字节流
            image_bytes = base64.b64decode(base64_str)
            # 转换为numpy数组
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            # 解码为图像（深度图用IMREAD_UNCHANGED保留16位数据）
            if is_depth:
                # 16位深度图解码（关键参数：cv2.IMREAD_UNCHANGED）
                image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
                if image.dtype != np.uint16:
                    print("警告：解码的深度图不是16位数据，可能存在格式错误")
            else:
                # 8位彩色图解码（BGR格式，OpenCV默认）
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"图像解码失败：{str(e)}")
            return None

    def request_both_images(self, save_local=True):
        """
        核心功能：请求服务端返回彩色图像+深度图像
        :param save_local: 是否保存到本地（True/False）
        :return: 图像字典（color_img, depth_img, timestamp）或None（失败）
        """
        try:
            # 发送GET请求获取图像（服务端同时支持POST，可替换为requests.post）

            response = requests.get(f"{self.base_url}/request_images", timeout=10)


            if response.status_code != 200:
                print(f"获取图像失败，HTTP状态码：{response.status_code}")
                return None

            result = response.json()
            if not result['success']:
                print(f"服务端返回错误：{result['error']}")
                return None

            # 1. 解码彩色图像（8位BGR）
            color_img = self.decode_base64_to_image(
                result['color_image']['data_base64'],
                is_depth=False
            )
            # 2. 解码深度图像（16位，单位毫米）
            depth_img = self.decode_base64_to_image(
                result['depth_image']['data_base64'],
                is_depth=True
            )

            if color_img is None or depth_img is None:
                print("图像解码失败，无法获取有效图像")
                return None

            # 3. 本地保存（可选）
            if save_local:
                timestamp = result['request_timestamp'].replace(":", "").replace(" ", "_").replace(".", "_")
                print('图像服务返回数据：', timestamp)
                # 保存彩色图
                color_save_path = os.path.join(self.save_dir, f"color_{timestamp}.png")
                cv2.imwrite(color_save_path, color_img)
                # 保存深度图（用无压缩格式，保留16位数据）
                depth_save_path = os.path.join(self.save_dir, f"depth_16bit_{timestamp}.png")
                cv2.imwrite(depth_save_path, depth_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                print(f"\n图像已保存至本地：")
                print(f"  彩色图：{color_save_path}")
                print(f"  深度图：{depth_save_path}（16位，单位毫米）")

            # 4. 验证深度数据（可选，打印深度统计信息）
            valid_depth = depth_img[depth_img > 0]  # 排除0值（无效深度）
            if len(valid_depth) > 0:
                print(f"\n深度数据验证（单位：毫米）：")
                print(f"  有效深度像素数：{len(valid_depth)}")
                print(f"  深度范围：{valid_depth.min()} ~ {valid_depth.max()}")
                print(f"  平均深度：{valid_depth.mean():.1f}")

            return {
                "color_img": color_img,
                "depth_img": depth_img,
                "timestamp": result['request_timestamp'],
                "resolution": f"{color_img.shape[1]}x{color_img.shape[0]}"
            }

        except Exception as e:
            print(f"获取双图像出错：{str(e)}")
            return None

    def show_images(self, color_img, depth_img, window_title="D435 相机图像"):
        """
        实时显示彩色图像和深度图像（深度图归一化便于可视化）
        :param color_img: 彩色图像数组
        :param depth_img: 深度图像数组（16位）
        :param window_title: 窗口标题
        """
        if color_img is None or depth_img is None:
            print("无法显示图像：输入图像为空")
            return

        # 深度图归一化（0-65535 → 0-255），便于可视化
        valid_depth = depth_img[depth_img > 0]
        if len(valid_depth) > 0:
            # 用有效深度范围归一化（避免0值影响对比度）
            depth_norm = cv2.normalize(
                depth_img, None, 0, 255,
                cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # 彩色映射（可选：JET色系，更直观）
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        else:
            depth_colormap = np.zeros_like(color_img)  # 无有效深度时显示黑色

        # 拼接图像（左右布局）
        combined_img = np.hstack((color_img, depth_colormap))
        # 显示窗口（按ESC键关闭）
        cv2.imshow(window_title, combined_img)
        print("\n提示：按 ESC 键关闭图像显示窗口")
        while True:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                cv2.destroyAllWindows()
                break


# ---------------------- 客户端使用示例 ----------------------
if __name__ == "__main__":
    # 1. 初始化客户端（需替换为你的服务端IP！）
    # 服务端IP可从服务端日志获取，如 "192.168.1.231"（默认端口11225）
    # client = D435CameraClient(server_ip="192.168.1.226", server_port=11228)  # camera1
    client = D435CameraClient(server_ip="192.168.1.231", server_port=11225)  # camera2

    # 2. 第一步：获取服务端状态，检查相机是否就绪
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
                exit(1)
        else:
            print("相机重启失败，程序退出")
            exit(1)

    # 3. 第二步：请求并获取彩色+深度图像
    print("\n=== 开始请求图像 ===")
    image_data = client.request_both_images(save_local=False)  # save_local=True 保存到本地
    if not image_data:
        print("获取图像失败，程序退出")
        exit(1)

    # 4. 第三步：显示图像（可选，按ESC关闭）
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