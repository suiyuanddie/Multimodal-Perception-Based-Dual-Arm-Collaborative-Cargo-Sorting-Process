import requests
import json
from typing import Dict, Optional, Any
import time
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


class RobotAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def connect(self, ip: str, port: int, timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}/connect"
        data = {
            "ip": ip,
            "port": port,
            "timeout": timeout
        }

        try:
            response = self.session.post(url, data=json.dumps(data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"连接请求失败: {str(e)}"}

    def disconnect(self) -> Dict[str, Any]:
        url = f"{self.base_url}/disconnect"

        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"断开连接请求失败: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        url = f"{self.base_url}/status"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"获取状态请求失败: {str(e)}"}

    def get_tcp_pose(self) -> Dict[str, Any]:
        url = f"{self.base_url}/get_tcp_pose"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"获取TCP位姿请求失败: {str(e)}"}

    def run_point(self,
                  pose: Dict[str, float],
                  speed: int,
                  tolerance: Optional[Dict[str, float]] = None,
                  timeout: int = 15) -> Dict[str, Any]:
        url = f"{self.base_url}/run_point"
        data = {
            "pose": pose,
            "speed": speed,
            "timeout": timeout
        }

        if tolerance:
            data["tolerance"] = tolerance

        try:
            response = self.session.post(url, data=json.dumps(data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"PTP运动请求失败: {str(e)}"}

    def run_line(self,
                 pose: Dict[str, float],
                 speed: int,
                 tolerance: Optional[Dict[str, float]] = None,
                 timeout: int = 15) -> Dict[str, Any]:
        url = f"{self.base_url}/run_line"
        data = {
            "pose": pose,
            "speed": speed,
            "timeout": timeout
        }

        if tolerance:
            data["tolerance"] = tolerance

        try:
            response = self.session.post(url, data=json.dumps(data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"直线运动请求失败: {str(e)}"}

    def shift_point(self,
                    x: float = 0,
                    y: float = 0,
                    z: float = 0,
                    r: float = 0,
                    p: float = 0,
                    _y_: float = 0,
                    speed: int = 50,
                    tolerance: Optional[Dict[str, float]] = None,
                    timeout: int = 15) -> Dict[str, Any]:
        url = f"{self.base_url}/shift_point"
        data = {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "p": p,
            "_y_": _y_,
            "speed": speed,
            "timeout": timeout
        }

        if tolerance:
            data["tolerance"] = tolerance

        try:
            response = self.session.post(url, data=json.dumps(data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"偏移运动请求失败: {str(e)}"}


class GripperAPIClient:
    def __init__(self, server_ip, server_port=11223):
        self.base_url = f"http://{server_ip}:{server_port}/gripper"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def init_gripper(self, port='/dev/armjaw', baud_rate=115200):
        url = f"{self.base_url}/init"
        payload = {
            "port": port,
            "baud_rate": baud_rate
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print(f"夹爪初始化成功: {result.get('message')}")
                print(f"串口: {result.get('port')}, 波特率: {result.get('baud_rate')}")
                return True
            else:
                print(f"夹爪初始化失败: {result.get('message')}")
                return False

        except requests.exceptions.ConnectTimeout:
            print("连接超时，请检查服务是否启动")
            return False
        except requests.exceptions.ConnectionError:
            print("连接失败，请检查IP和端口是否正确")
            return False
        except Exception as e:
            print(f"初始化发生错误: {str(e)}")
            return False

    def gripper_catch(self):
        url = f"{self.base_url}/catch"

        try:
            response = self.session.post(url, json={}, timeout=5)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print(f"夹取成功: {result.get('message')}")
                return True
            else:
                print(f"夹取失败: {result.get('message')}")
                return False

        except Exception as e:
            print(f"夹取发生错误: {str(e)}")
            return False

    def gripper_release(self):
        url = f"{self.base_url}/release"

        try:
            response = self.session.post(url, json={}, timeout=5)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print(f"释放成功: {result.get('message')}")
                return True
            else:
                print(f"释放失败: {result.get('message')}")
                return False

        except Exception as e:
            print(f"释放发生错误: {str(e)}")
            return False

    def get_status(self):
        url = f"{self.base_url}/status"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            status = response.json()

            print("\n当前夹爪状态:")
            print(f"是否初始化: {'是' if status.get('initialized') else '否'}")
            print(f"串口端口: {status.get('serial_port')}")
            print(f"波特率: {status.get('baud_rate')}")
            print(f"串口是否打开: {'是' if status.get('serial_open') else '否'}")
            print(f"最后操作时间: {status.get('last_operation')}")

            return status

        except Exception as e:
            print(f"获取状态发生错误: {str(e)}")
            return None

    def deinit_gripper(self):
        url = f"{self.base_url}/deinit"

        try:
            response = self.session.post(url, json={}, timeout=5)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print(f"关闭成功: {result.get('message')}")
                return True
            else:
                print(f"关闭失败: {result.get('message')}")
                return False

        except Exception as e:
            print(f"关闭发生错误: {str(e)}")
            return False


class ROSImageClient:
    def __init__(self, server_ip="localhost", server_port=11223):
        self.base_url = f"http://{server_ip}:{server_port}/ros/image"
        self.session = requests.Session()
        self.session.timeout = 10

    def get_images(self, include_color: bool = True, include_depth: bool = True) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/get"
        params = {
            "include_color": str(include_color).lower(),
            "include_depth": str(include_depth).lower()
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                return result.get("images", {})
            else:
                print(f"获取图像失败：{result.get('message', '未知错误')}")
                return None

        except requests.exceptions.ConnectTimeout:
            print("获取图像超时：服务端未响应")
            return None
        except requests.exceptions.ConnectionError:
            print("获取图像连接失败：请检查服务端IP和端口")
            return None
        except Exception as e:
            print(f"获取图像异常：{str(e)}")
            return None

    def get_service_status(self) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/status"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            status = response.json()
            print(f"\n=== ROS图像服务状态 ===")
            print(f"服务是否运行：{'是' if status.get('is_running') else '否'}")
            print(f"彩色图更新次数：{status.get('color_update_count', 0)}")
            print(f"深度图更新次数：{status.get('depth_update_count', 0)}")
            print(f"彩色图话题：{status.get('color_topic', '未知')}")
            print(f"深度图话题：{status.get('depth_topic', '未知')}")
            return status
        except Exception as e:
            print(f"获取服务状态失败：{str(e)}")
            return None


def decode_color_image(base64_data: str) -> Optional[np.ndarray]:
    if not base64_data:
        print("解码失败：彩色图Base64数据为空")
        return None

    try:
        image_bytes = base64.b64decode(base64_data)
        pil_img = Image.open(BytesIO(image_bytes))
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv_img
    except base64.binascii.Error:
        print("解码失败：Base64数据格式错误")
        return None
    except Exception as e:
        print(f"解码彩色图像异常：{str(e)}")
        return None


def process_depth_image(depth_data: list, width: int, height: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not depth_data or width <= 0 or height <= 0:
        print("处理失败：深度图数据或尺寸无效")
        return None, None

    try:
        depth_array = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)

        if np.max(depth_array) > np.min(depth_array):
            depth_normalized = cv2.normalize(
                depth_array, None, 0, 255,
                cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            )
        else:
            depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)

        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return depth_array, depth_colored

    except Exception as e:
        print(f"处理深度图像异常：{str(e)}")
        return None, None


if __name__ == "__main__":
    SERVER_IP = "192.168.1.226"
    SERVER_PORT = 11223
    BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

    # 机器人控制
    robot_client = RobotAPIClient(BASE_URL)
    try:
        print("=== 1. 机器人控制 ===")
        connect_result = robot_client.connect("169.254.128.88", 8080)
        print(f"连接结果：{connect_result}")
        if not connect_result.get("success"):
            raise Exception("机器人连接失败，跳过后续控制逻辑")

        print(f"连接状态：{robot_client.get_status()}")
        print(f"当前TCP位姿：{robot_client.get_tcp_pose()}")

        # target_pose = {
        #     "x": 275.0, "y": -443.0, "z": 717.0,
        #     "r": -90.0, "p": -1.0, "_y_": -138.0
        # }
        target_pose = {
            "x": -129.0, "y": -12.0, "z": 915.0,
            "r": -89.0, "p": -1.0, "_y_": -135.0
        }
        # target_pose = {
        #     "x": 184,
        #     "y": -135.5,
        #     "z": 647.0,
        #     "r": -160,
        #     "p": -0.1,
        #     "_y_": -99.0
        # }
        ptp_result = robot_client.run_point(
            pose=target_pose, speed=10, tolerance={"x": 1.5, "y": 1.5}
        )
        print(ptp_result)
        print(f"PTP运动结果：{'成功' if ptp_result['verification']['success'] else '失败'}")
        print(f"运动误差：{ptp_result['verification']['errors']}")

        a=robot_client.get_tcp_pose()
        print(a)

    except Exception as e:
        print(f"机器人控制异常：{str(e)}")
    finally:
        print(f"断开机器人连接：{robot_client.disconnect()}")

    # 夹爪控制
    gripper_client = GripperAPIClient(SERVER_IP, SERVER_PORT)
    try:
        print("\n=== 2. 夹爪控制 ===")
        if not gripper_client.init_gripper():
            raise Exception("夹爪初始化失败，跳过后续夹爪逻辑")

        time.sleep(2)
        gripper_client.get_status()

        print("\n执行夹取...")
        gripper_client.gripper_catch()
        time.sleep(1)

        print("\n执行释放...")
        gripper_client.gripper_release()
        time.sleep(1)

    except Exception as e:
        print(f"夹爪控制异常：{str(e)}")
    finally:
        gripper_client.deinit_gripper()

    # # 图像读取
    # print("\n=== 3. 实时图像读取 ===")
    # image_client = ROSImageClient(SERVER_IP, SERVER_PORT)
    #
    # service_status = image_client.get_service_status()
    # if not service_status or not service_status.get("is_running"):
    #     print("警告：ROS图像服务未运行，可能无法获取最新图像")
    #     time.sleep(2)
    #
    # try:
    #     print("\n开始读取图像（按'q'退出）...")
    #     while True:
    #         images = image_client.get_images(include_color=True, include_depth=True)
    #         if not images:
    #             time.sleep(0.5)
    #             continue
    #
    #         # 处理彩色图像
    #         if "color" in images and images["color"]["available"]:
    #             color_img = decode_color_image(images["color"]["base64_data"])
    #             if color_img is not None:
    #                 color_img_resized = cv2.resize(color_img, (640, 480))
    #                 # cv2.imshow("Real-Time Color Image", color_img_resized)
    #                 cv2.imwrite('output/color.png', color_img_resized)
    #
    #         # 处理深度图像（增加错误处理）
    #         if "depth" in images and images["depth"]["available"]:
    #             try:
    #                 # 从深度图像数据中提取参数，增加默认值处理
    #                 depth_data = images["depth"]["data"]
    #                 # 尝试从服务状态获取尺寸，如果没有则使用默认值
    #                 depth_width = images["depth"].get("width") or service_status.get("depth_width", 640)
    #                 depth_height = images["depth"].get("height") or service_status.get("depth_height", 480)
    #
    #                 # 验证尺寸有效性
    #                 if depth_width <= 0 or depth_height <= 0:
    #                     print(f"无效的深度图尺寸: {depth_width}x{depth_height}，使用默认尺寸640x480")
    #                     depth_width, depth_height = 640, 480
    #
    #                 depth_array, depth_colored = process_depth_image(
    #                     depth_data, depth_width, depth_height
    #                 )
    #                 cv2.imwrite('output/depth.png', depth_array)
    #
    #                 if depth_colored is not None:
    #                     depth_colored_resized = cv2.resize(depth_colored, (640, 480))
    #                     cv2.imshow("Real-Time Depth Image (Colored)", depth_colored_resized)
    #
    #
    #             except KeyError as e:
    #                 print(f"深度图像数据缺少必要字段: {e}，跳过深度图显示")
    #             except Exception as e:
    #                 print(f"处理深度图像时出错: {e}，跳过深度图显示")
    #
    #         # 按键控制
    #         key = cv2.waitKey(1)
    #         if key == ord('q'):
    #             print("用户按下'q'，退出图像读取")
    #             break
    #         time.sleep(0.1)
    #
    # except KeyboardInterrupt:
    #     print("\n用户中断图像读取")
    # finally:
    #     cv2.destroyAllWindows()
    #     print("图像读取结束，资源已清理")
