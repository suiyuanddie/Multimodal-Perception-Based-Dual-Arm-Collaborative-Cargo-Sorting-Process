import socket
import time
import threading
import serial  # 新增：夹爪串口通信依赖
import select  # 新增：夹爪超时处理依赖
from flask import Flask, request, jsonify


app = Flask(__name__)


# --------------------------- 原有机器人位姿结构体 ---------------------------
class Pose:
    """机械臂末端TCP位姿结构体"""
    def __init__(self, x=0.0, y=0.0, z=0.0, r=0.0, p=0.0, _y_=0.0):
        self.x = x       # X坐标
        self.y = y       # Y坐标
        self.z = z       # Z坐标
        self.r = r       # 旋转角度r
        self.p = p       # 旋转角度p
        self._y_ = _y_   # 旋转角度y

class JPose:
    """机械臂关节角度结构体"""
    def __init__(self, j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0):
        self.j1 = j1     # 关节1角度
        self.j2 = j2     # 关节2角度
        self.j3 = j3     # 关节3角度
        self.j4 = j4     # 关节4角度
        self.j5 = j5     # 关节5角度
        self.j6 = j6     # 关节6角度


# --------------------------- 新增：夹爪控制器类 ---------------------------
class GripperController:
    def __init__(self, port='/dev/armjaw', baud_rate=115200):
        self.serial_port = serial.Serial()
        self.serial_port.port = port
        self.serial_port.baudrate = baud_rate
        # 显式设置串口参数（与硬件匹配）
        self.serial_port.bytesize = serial.EIGHTBITS
        self.serial_port.parity = serial.PARITY_NONE
        self.serial_port.stopbits = serial.STOPBITS_ONE
        self.serial_port.timeout = 0  # 非阻塞模式
        self.lock = threading.Lock()  # 线程锁：保护串口并发操作
        self.response_received = False

    def serial_init(self):
        """初始化串口连接 + 夹爪参数初始化"""
        try:
            if not self.serial_port.is_open:
                self.serial_port.open()
            print("Gripper serial port opened:", self.serial_port.port)
            # 串口打开后，执行夹爪硬件初始化
            return self.grap_init()
        except Exception as e:
            print(f"Gripper serial init failed: {e}")
            return False

    def grap_init(self):
        """夹爪硬件初始化（发送初始化指令并验证响应）"""
        init_data = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x01, 0x24, 0x78])
        try:
            with self.lock:
                # 发送初始化指令（检查发送完整性）
                sent_bytes = self.serial_port.write(init_data)
                if sent_bytes != len(init_data):
                    print(f"Gripper init send failed (sent {sent_bytes}/{len(init_data)} bytes)")
                    return False

                # 使用select实现1秒超时等待响应
                ready, _, _ = select.select([self.serial_port], [], [], 1.0)
                if not ready:
                    print("Gripper init timeout (no response)")
                    return False

                # 读取8字节响应（夹爪标准响应长度）
                response = self.serial_port.read(8)
                if len(response) != 8:
                    print(f"Gripper init response invalid (got {len(response)}/8 bytes)")
                    return False

            print(f"Gripper init success, response: {[hex(b) for b in response]}")
            return True
        except Exception as e:
            print(f"Gripper init error: {e}")
            return False

    def grap_catch(self):
        """夹爪夹取动作（线程安全）"""
        catch_data = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x09, 0xFF, 0x00, 0xFF, 0xFF, 0x9E, 0x95])
        try:
            with self.lock:  # 确保多线程下串口操作唯一
                if not self.serial_port.is_open:
                    raise Exception("Serial port is closed")

                sent_bytes = self.serial_port.write(catch_data)
                if sent_bytes != len(catch_data):
                    raise Exception(f"Send failed (sent {sent_bytes}/{len(catch_data)} bytes)")

            print("Gripper catch command sent successfully")
        except Exception as e:
            print(f"Gripper catch failed: {e}")
            raise  # 抛出异常，让API层捕获并返回错误

    def grap_release(self):
        """夹爪释放动作（线程安全）"""
        release_data = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x09, 0x00, 0x00, 0xFF, 0xFF, 0xAE, 0x81])
        try:
            with self.lock:  # 确保多线程下串口操作唯一
                if not self.serial_port.is_open:
                    raise Exception("Serial port is closed")

                sent_bytes = self.serial_port.write(release_data)
                if sent_bytes != len(release_data):
                    raise Exception(f"Send failed (sent {sent_bytes}/{len(release_data)} bytes)")

            print("Gripper release command sent successfully")
        except Exception as e:
            print(f"Gripper release failed: {e}")
            raise  # 抛出异常，让API层捕获并返回错误

    def serial_deinit(self):
        """关闭夹爪串口（安全释放资源）"""
        try:
            if self.serial_port.is_open:
                self.serial_port.close()
                print(f"Gripper serial port closed: {self.serial_port.port}")
        except Exception as e:
            print(f"Gripper serial deinit error: {e}")


# --------------------------- 原有机器人控制器类 ---------------------------
class RobotController:
    def __init__(self):
        self.count = 0
        self.sock = None
        self.count_lock = threading.Lock()
        # 默认阈值（毫米/度）
        self.default_tolerance = {
            'x': 1.0, 'y': 1.0, 'z': 1.0,
            'r': 1.0, 'p': 1.0, '_y_': 1.0,
        }

    def connect_socket(self, ip='127.0.0.1', port=8080, timeout=10):
        """建立socket连接（带超时）"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((ip, port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(None)
            print("Robot socket connected successfully.")
            return True
        except socket.timeout:
            print(f"Robot socket timeout (>={timeout}s)")
            return False
        except ConnectionRefusedError:
            print("Robot connection refused: check IP/port")
            return False
        except Exception as e:
            print(f"Robot socket connect failed: {e}")
            return False

    def split_res(self, res, index, dest):
        """分割响应字符串"""
        tokens = res.split('III')
        if 0 < index <= len(tokens):
            dest[0] = tokens[index-1]
            return 0
        return -1

    def _recv_complete(self, sock, end_marker=b"III/b/f"):
        """循环接收直到获取完整响应"""
        buf = b""
        while True:
            try:
                chunk = sock.recv(1024)
                if not chunk:
                    return None
                buf += chunk
                if end_marker in buf:
                    return buf.decode()
                if len(buf) > 1024*1024:
                    print("Robot response too large (>1MB), truncating")
                    return buf.decode()
            except ConnectionResetError:
                print("Robot disconnected (reset)")
                return None
            except Exception as e:
                print(f"Robot recv error: {e}")
                return None

    def get_joint_pose(self, sock, c: Pose, j: JPose):
        """获取关节角度"""
        with self.count_lock:
            self.count += 1
            current_count = self.count

        cmd_val = f"GetInverseKin(0,{c.x:.4f},{c.y:.4f},{c.z:.4f},{c.r:.4f},{c.p:.4f},{c._y_:.4f},-1)"
        cmd = f"/f/bIII{(current_count % 100) + 100}III377III{len(cmd_val)}III{cmd_val}III/b/f"
        print(f"Sending joint pose cmd: {cmd}")

        try:
            send_num = sock.send(cmd.encode())
            if send_num != len(cmd):
                print(f"Robot send failed (sent {send_num}/{len(cmd)})")
                return -1

            buf = self._recv_complete(sock)
            if not buf:
                print("No response for joint pose")
                return -1

            valid = [""]
            if self.split_res(buf, 3, valid) != 0:
                return -1

            if int(valid[0]) != 377:
                print(f"Joint pose invalid: {buf}")
                return -1

            pose_data = [""]
            if self.split_res(buf, 5, pose_data) == 0:
                try:
                    j1, j2, j3, j4, j5, j6 = map(float, pose_data[0].split(','))
                    j.j1, j.j2, j.j3, j.j4, j.j5, j.j6 = j1, j2, j3, j4, j5, j6
                    return 0
                except ValueError:
                    print(f"Invalid joint data: {pose_data[0]}")
                    return -1
            else:
                print("Joint pose parse failed")
                return -1

        except Exception as e:
            print(f"Joint pose error: {e}")
            return -1

    def get_tcp_pose(self, sock, cur_pose: Pose):
        """获取TCP位姿"""
        with self.count_lock:
            self.count += 1
            current_count = self.count

        cmd_val = f"/f/bIII{(current_count % 100) + 100}III377III18IIIGetActualTCPPose()III/b/f"
        print(f"Sending TCP pose cmd: {cmd_val}")

        try:
            send_num = sock.send(cmd_val.encode())
            if send_num != len(cmd_val):
                print(f"Robot send failed (sent {send_num}/{len(cmd_val)})")
                return -1

            buf = self._recv_complete(sock)
            if not buf:
                print("No response for TCP pose")
                return -1

            valid = [""]
            if self.split_res(buf, 3, valid) != 0:
                return -1

            if int(valid[0]) != 377:
                print(f"TCP pose invalid: {buf}")
                return -1

            pose_data = [""]
            if self.split_res(buf, 5, pose_data) == 0:
                try:
                    x, y, z, r, p, y_ = map(float, pose_data[0].split(','))
                    cur_pose.x, cur_pose.y, cur_pose.z = x, y, z
                    cur_pose.r, cur_pose.p, cur_pose._y_ = r, p, y_
                    return 0
                except ValueError:
                    print(f"Invalid TCP data: {pose_data[0]}")
                    return -1
            else:
                return -1

        except Exception as e:
            print(f"TCP pose error: {e}")
            return -1

    def wait_for_move_complete(self, sock, target_offset, original_pose,
                             tolerance=None, timeout=10):
        """带可输入阈值的到位验证函数"""
        check_tolerance = self.default_tolerance.copy()
        if tolerance:
            check_tolerance.update(tolerance)

        # 计算目标位姿
        target = Pose(
            x=original_pose.x + target_offset[0],
            y=original_pose.y + target_offset[1],
            z=original_pose.z + target_offset[2],
            r=original_pose.r + target_offset[3],
            p=original_pose.p + target_offset[4],
            _y_=original_pose._y_ + target_offset[5]
        )

        start_time = time.time()
        stable_count = 0
        required_stable = 2  # 连续稳定次数

        while time.time() - start_time < timeout:
            current = Pose()
            if self.get_tcp_pose(sock, current) != 0:
                time.sleep(0.5)
                continue

            # 计算各轴误差
            errors = {
                'x': abs(current.x - target.x),
                'y': abs(current.y - target.y),
                'z': abs(current.z - target.z),
                'r': abs(current.r - target.r),
                'p': abs(current.p - target.p),
                '_y_': abs(current._y_ - target._y_)
            }

            # 检查是否在阈值范围内
            all_in_tolerance = all(
                errors[axis] <= check_tolerance[axis]
                for axis in errors
            )

            if all_in_tolerance:
                stable_count += 1
                if stable_count >= required_stable:
                    elapsed = round(time.time() - start_time, 2)
                    return {
                        "success": True,
                        "current_pose": {k: v for k, v in vars(current).items()},
                        "target_pose": {k: v for k, v in vars(target).items()},
                        "errors": errors,
                        "tolerance_used": check_tolerance,
                        "elapsed_time": elapsed,
                        "message": f"已稳定在阈值范围内（连续{required_stable}次检测）"
                    }
            else:
                stable_count = 0

            time.sleep(0.3)

        # 超时处理：详细返回目标、实际位置和误差
        final = Pose()
        self.get_tcp_pose(sock, final)
        final_errors = {k: abs(getattr(final, k) - getattr(target, k)) for k in vars(final).keys()}
        return {
            "success": False,
            "current_pose": {k: v for k, v in vars(final).items()},
            "target_pose": {k: v for k, v in vars(target).items()},
            "errors": final_errors,
            "tolerance_used": check_tolerance,
            "elapsed_time": round(time.time() - start_time, 2),
            "message": f"超时({timeout}秒)，未达到阈值范围"
        }

    def run_point(self, sock, p: Pose, speed):
        """PTP运动（弧线）- 原始函数"""
        j = JPose()
        res = self.get_joint_pose(sock, p, j)
        time.sleep(0.05)

        if res == 0:
            with self.count_lock:
                self.count += 1
                current_count = self.count

            cmd_val = (f"MoveJ({j.j1:.3f},{j.j2:.3f},{j.j3:.3f},{j.j4:.3f},{j.j5:.3f},{j.j6:.3f},"
                      f"{p.x:.3f},{p.y:.3f},{p.z:.3f},{p.r:.3f},{p.p:.3f},{p._y_:.3f}, "
                      f"0,0, {speed % 100},100,100,0.000,0.000,0.000,0.000,-1,0,0,0,0,0,0,0)")
            cmd = f"/f/bIII{(current_count % 100) + 100}III201III{len(cmd_val)}III{cmd_val}III/b/f"
            print(f"Sending run point cmd: {cmd}")

            try:
                send_num = sock.send(cmd.encode())
                if send_num != len(cmd):
                    print(f"Robot send failed (sent {send_num}/{len(cmd)})")
                    return -1

                buf = self._recv_complete(sock)
                if not buf:
                    print("No response for run point")
                    return -1

                valid = [""]
                self.split_res(buf, 3, valid)
                if int(valid[0]) != 201:
                    print(f"Run point invalid: {buf}")
                    return -1

                time.sleep(0.05)
                return 0
            except Exception as e:
                print(f"Run point error: {e}")
                return -1
        else:
            print("Get joint pose failed in run_point")
            return -1

    def run_point_with_safety(self, sock, p: Pose, speed, tolerance=None, timeout=15):
        """带到位判断的PTP运动（弧线）"""
        # 获取原始位姿
        original_pose = Pose()
        if self.get_tcp_pose(sock, original_pose) != 0:
            return {"success": False, "message": "获取初始位姿失败"}

        # 计算目标偏移量（目标位姿 - 原始位姿）
        target_offset = tuple(getattr(p, k) - getattr(original_pose, k) for k in vars(p).keys())

        # 执行移动命令
        move_start = time.time()
        move_res = self.run_point(sock, p, speed)
        if move_res != 0:
            return {
                "success": False,
                "message": "PTP移动命令被机械臂拒绝",
                "original_pose": {k: v for k, v in vars(original_pose).items()},
                "target_pose": {k: v for k, v in vars(p).items()}
            }

        # 等待到位验证
        verification = self.wait_for_move_complete(
            sock, target_offset=target_offset, original_pose=original_pose,
            tolerance=tolerance, timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": {k: v for k, v in vars(original_pose).items()},
            "target_pose": {k: v for k, v in vars(p).items()}
        }

    def run_point_line(self, sock, c_p: Pose, speed):
        """直线运动 - 原始函数"""
        j = JPose()
        res = self.get_joint_pose(sock, c_p, j)
        time.sleep(0.05)

        if res == 0:
            with self.count_lock:
                self.count += 1
                current_count = self.count

            cmd_val = (f"MoveL({j.j1:.3f},{j.j2:.3f},{j.j3:.3f},{j.j4:.3f},{j.j5:.3f},{j.j6:.3f},"
                      f"{c_p.x:.3f},{c_p.y:.3f},{c_p.z:.3f},{c_p.r:.3f},{c_p.p:.3f},{c_p._y_:.3f}, "
                      f"0,0, {speed % 100},100,100,-1 ,0.000,0.000,0.000,0.000,0,2,0,0,0,0,0,0)")
            cmd = f"/f/bIII{(current_count % 100) + 100}III203III{len(cmd_val)}III{cmd_val}III/b/f"
            print(f"Sending line cmd: {cmd}")

            try:
                send_num = sock.send(cmd.encode())
                if send_num != len(cmd):
                    print(f"Robot send failed (sent {send_num}/{len(cmd)})")
                    return -1

                buf = self._recv_complete(sock)
                if not buf:
                    print("No response for line move")
                    return -1

                valid = [""]
                self.split_res(buf, 3, valid)
                if int(valid[0]) != 203:
                    print(f"Line move invalid: {buf}")
                    return -1

                time.sleep(0.05)
                return 0
            except Exception as e:
                print(f"Line move error: {e}")
                return -1
        else:
            print("Get joint pose failed in run_point_line")
            return -1

    def run_point_line_with_safety(self, sock, c_p: Pose, speed, tolerance=None, timeout=15):
        """带到位判断的直线运动"""
        # 获取原始位姿
        original_pose = Pose()
        if self.get_tcp_pose(sock, original_pose) != 0:
            return {"success": False, "message": "获取初始位姿失败"}

        # 计算目标偏移量（目标位姿 - 原始位姿）
        target_offset = tuple(getattr(c_p, k) - getattr(original_pose, k) for k in vars(c_p).keys())

        # 执行移动命令
        move_start = time.time()
        move_res = self.run_point_line(sock, c_p, speed)
        if move_res != 0:
            return {
                "success": False,
                "message": "直线移动命令被机械臂拒绝",
                "original_pose": {k: v for k, v in vars(original_pose).items()},
                "target_pose": {k: v for k, v in vars(c_p).items()}
            }

        # 等待到位验证
        verification = self.wait_for_move_complete(
            sock, target_offset=target_offset, original_pose=original_pose,
            tolerance=tolerance, timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": {k: v for k, v in vars(original_pose).items()},
            "target_pose": {k: v for k, v in vars(c_p).items()}
        }

    def shift_point_with_safety(self, sock, x, y, z, r=0, p=0, _y_=0, speed=50, tolerance=None, timeout=15):
        """带安全机制的偏移运动"""
        original = Pose()
        if self.get_tcp_pose(sock, original) != 0:
            return {"success": False, "message": "获取初始位姿失败"}

        move_start = time.time()
        move_res = self.shift_point(sock, x, y, z, r, p, _y_, speed)
        if move_res != 0:
            return {
                "success": False,
                "message": "偏移移动命令被机械臂拒绝",
                "original_pose": {k: v for k, v in vars(original).items()}
            }

        verification = self.wait_for_move_complete(
            sock, target_offset=(x, y, z, r, p, _y_), original_pose=original,
            tolerance=tolerance, timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": {k: v for k, v in vars(original).items()}
        }

    def shift_point(self, sock, x, y, z, r, p, _y_, speed):
        """原始偏移运动方法"""
        c_p = Pose()
        res = self.get_tcp_pose(sock, c_p)
        time.sleep(0.05)

        if res != 0:
            print("Get tcp pose failed in shift_point")
            return -1

        j = JPose()
        res = self.get_joint_pose(sock, c_p, j)
        time.sleep(0.05)

        if res == 0:
            with self.count_lock:
                self.count += 1
                current_count = self.count

            cmd_val = (f"MoveL({j.j1:.3f},{j.j2:.3f},{j.j3:.3f},{j.j4:.3f},{j.j5:.3f},{j.j6:.3f},"
                      f"{c_p.x:.3f},{c_p.y:.3f},{c_p.z:.3f},{c_p.r:.3f},{c_p.p:.3f},{c_p._y_:.3f}, "
                      f"0,0, {speed % 100},100,100,-1 ,0.000,0.000,0.000,0.000,0,2,"
                      f"{x:.3f},{y:.3f},{z:.3f},{r:.3f},{p:.3f},{_y_:.3f})")
            cmd = f"/f/bIII{(current_count % 100) + 100}III203III{len(cmd_val)}III{cmd_val}III/b/f"
            print(f"Sending shift cmd: {cmd}")

            try:
                send_num = sock.send(cmd.encode())
                if send_num != len(cmd):
                    print(f"Robot send failed (sent {send_num}/{len(cmd)})")
                    return -1

                buf = self._recv_complete(sock)
                if not buf:
                    print("No response for shift move")
                    return -1

                valid = [""]
                self.split_res(buf, 3, valid)
                if int(valid[0]) != 203:
                    print(f"Shift move invalid: {buf}")
                    return -1

                time.sleep(0.05)
                return 0
            except Exception as e:
                print(f"Shift move error: {e}")
                return -1
        else:
            print("Get joint pose failed in shift_point")
            return -1

    def close_socket(self):
        """关闭socket连接"""
        time.sleep(2)
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                print(e)
            self.sock.close()
            self.sock = None
            print("Robot socket closed.")


# --------------------------- 全局实例与状态管理（机器人+夹爪） ---------------------------
# 1. 机器人全局实例与状态
robot = RobotController()
robot_state_lock = threading.Lock()
robot_connection_state = {
    'connected': False,
    'ip': None,
    'port': None,
    'last_heartbeat': None
}

# 2. 夹爪全局实例与状态（新增）
gripper = GripperController()
gripper_state_lock = threading.Lock()
gripper_state = {
    'initialized': False,
    'port': '/dev/armjaw',  # 默认串口端口
    'baud_rate': 115200,    # 默认波特率
    'last_operation': None
}


# --------------------------- 原有机器人API路由 ---------------------------
@app.route('/connect', methods=['POST'])
def connect_robot():
    """建立与机器人的连接"""
    global robot_connection_state
    data = request.json

    if not data or 'ip' not in data or 'port' not in data:
        return jsonify({"success": False, "message": "请提供IP和端口"}), 400

    with robot_state_lock:
        if robot_connection_state['connected']:
            return jsonify({"success": False, "message": "已处于连接状态"}), 400

        ip = data['ip']
        port = int(data['port'])
        timeout = int(data.get('timeout', 10))

        success = robot.connect_socket(ip, port, timeout)
        if success:
            robot_connection_state = {
                'connected': True,
                'ip': ip,
                'port': port,
                'last_heartbeat': time.time()
            }
            return jsonify({
                "success": True,
                "message": "机器人连接成功",
                "ip": ip,
                "port": port
            })
        else:
            return jsonify({"success": False, "message": "机器人连接失败，请检查IP和端口"}), 500


@app.route('/disconnect', methods=['POST'])
def disconnect_robot():
    """断开与机器人的连接"""
    global robot_connection_state

    with robot_state_lock:
        if not robot_connection_state['connected']:
            return jsonify({"success": False, "message": "未处于连接状态"}), 400

        robot.close_socket()
        robot_connection_state['connected'] = False
        return jsonify({"success": True, "message": "机器人已断开连接"})


@app.route('/status', methods=['GET'])
def robot_status():
    """获取机器人当前连接状态"""
    global robot_connection_state

    with robot_state_lock:
        return jsonify({
            "connected": robot_connection_state['connected'],
            "ip": robot_connection_state['ip'],
            "port": robot_connection_state['port'],
            "last_heartbeat": robot_connection_state['last_heartbeat']
        })


@app.route('/get_tcp_pose', methods=['GET'])
def get_robot_tcp_pose():
    """获取机器人当前TCP位姿"""
    with robot_state_lock:
        if not robot_connection_state['connected']:
            return jsonify({"success": False, "message": "未连接到机器人"}), 400

        if not robot.sock:
            robot_connection_state['connected'] = False
            return jsonify({"success": False, "message": "机器人连接已断开"}), 500

        current_pose = Pose()
        res = robot.get_tcp_pose(robot.sock, current_pose)

        if res == 0:
            robot_connection_state['last_heartbeat'] = time.time()
            return jsonify({
                "success": True,
                "pose": {k: v for k, v in vars(current_pose).items()}
            })
        else:
            return jsonify({"success": False, "message": "获取机器人位姿失败"}), 500


@app.route('/run_point', methods=['POST'])
def robot_run_point():
    """执行机器人PTP运动（弧线）"""
    data = request.json

    if not data or 'pose' not in data or 'speed' not in data:
        return jsonify({"success": False, "message": "请提供目标位姿和速度"}), 400

    pose_data = data['pose']
    required_fields = ['x', 'y', 'z', 'r', 'p', '_y_']
    for field in required_fields:
        if field not in pose_data:
            return jsonify({"success": False, "message": f"位姿缺少必要参数: {field}"}), 400

    with robot_state_lock:
        if not robot_connection_state['connected']:
            return jsonify({"success": False, "message": "未连接到机器人"}), 400

        if not robot.sock:
            robot_connection_state['connected'] = False
            return jsonify({"success": False, "message": "机器人连接已断开"}), 500

        # 创建目标位姿
        target_pose = Pose(**{k: float(pose_data[k]) for k in required_fields})

        # 执行运动
        result = robot.run_point_with_safety(
            robot.sock, target_pose,
            speed=int(data['speed']),
            tolerance=data.get('tolerance'),
            timeout=int(data.get('timeout', 15))
        )

        robot_connection_state['last_heartbeat'] = time.time()
        return jsonify(result)


@app.route('/run_line', methods=['POST'])
def robot_run_line():
    """执行机器人直线运动"""
    data = request.json

    if not data or 'pose' not in data or 'speed' not in data:
        return jsonify({"success": False, "message": "请提供目标位姿和速度"}), 400

    pose_data = data['pose']
    required_fields = ['x', 'y', 'z', 'r', 'p', '_y_']
    for field in required_fields:
        if field not in pose_data:
            return jsonify({"success": False, "message": f"位姿缺少必要参数: {field}"}), 400

    with robot_state_lock:
        if not robot_connection_state['connected']:
            return jsonify({"success": False, "message": "未连接到机器人"}), 400

        if not robot.sock:
            robot_connection_state['connected'] = False
            return jsonify({"success": False, "message": "机器人连接已断开"}), 500

        # 创建目标位姿
        target_pose = Pose(**{k: float(pose_data[k]) for k in required_fields})

        # 执行运动
        result = robot.run_point_line_with_safety(
            robot.sock, target_pose,
            speed=int(data['speed']),
            tolerance=data.get('tolerance'),
            timeout=int(data.get('timeout', 15))
        )

        robot_connection_state['last_heartbeat'] = time.time()
        return jsonify(result)


@app.route('/shift_point', methods=['POST'])
def robot_shift_point():
    """执行机器人偏移运动"""
    data = request.json

    if not data or 'speed' not in data:
        return jsonify({"success": False, "message": "请提供偏移量和速度"}), 400

    with robot_state_lock:
        if not robot_connection_state['connected']:
            return jsonify({"success": False, "message": "未连接到机器人"}), 400

        if not robot.sock:
            robot_connection_state['connected'] = False
            return jsonify({"success": False, "message": "机器人连接已断开"}), 500

        # 执行偏移运动
        result = robot.shift_point_with_safety(
            robot.sock,
            x=float(data.get('x', 0)),
            y=float(data.get('y', 0)),
            z=float(data.get('z', 0)),
            r=float(data.get('r', 0)),
            p=float(data.get('p', 0)),
            _y_=float(data.get('_y_', 0)),
            speed=int(data['speed']),
            tolerance=data.get('tolerance'),
            timeout=int(data.get('timeout', 15))
        )

        robot_connection_state['last_heartbeat'] = time.time()
        return jsonify(result)


# --------------------------- 新增：夹爪API路由 ---------------------------
@app.route('/gripper/init', methods=['POST'])
def gripper_init():
    """初始化夹爪（打开串口+硬件初始化）"""
    global gripper, gripper_state
    data = request.json or {}

    with gripper_state_lock:
        # 检查是否已初始化
        if gripper_state['initialized']:
            return jsonify({"success": False, "message": "夹爪已初始化，无需重复操作"}), 400

        # 可选：从请求中更新串口参数（默认使用全局配置）
        port = data.get('port', gripper_state['port'])
        baud_rate = int(data.get('baud_rate', gripper_state['baud_rate']))

        # 更新夹爪串口配置
        gripper.serial_port.port = port
        gripper.serial_port.baudrate = baud_rate
        gripper_state['port'] = port
        gripper_state['baud_rate'] = baud_rate

        # 执行初始化（串口打开 + 硬件初始化）
        init_success = gripper.serial_init()
        if init_success:
            gripper_state['initialized'] = True
            gripper_state['last_operation'] = time.time()
            return jsonify({
                "success": True,
                "message": "夹爪初始化成功",
                "port": port,
                "baud_rate": baud_rate
            })
        else:
            return jsonify({"success": False, "message": "夹爪初始化失败（检查串口/硬件）"}), 500


@app.route('/gripper/catch', methods=['POST'])
def gripper_catch():
    """执行夹爪夹取动作"""
    global gripper, gripper_state

    with gripper_state_lock:
        # 前置检查：是否已初始化 + 串口是否正常
        if not gripper_state['initialized']:
            return jsonify({"success": False, "message": "夹爪未初始化，请先调用/gripper/init"}), 400
        if not gripper.serial_port.is_open:
            gripper_state['initialized'] = False  # 重置状态
            return jsonify({"success": False, "message": "夹爪串口已断开，请重新初始化"}), 500

        # 执行夹取动作
        try:
            gripper.grap_catch()
            gripper_state['last_operation'] = time.time()
            return jsonify({
                "success": True,
                "message": "夹取指令已发送",
                "last_operation": gripper_state['last_operation']
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"夹取失败: {str(e)}"}), 500


@app.route('/gripper/release', methods=['POST'])
def gripper_release():
    """执行夹爪释放动作"""
    global gripper, gripper_state

    with gripper_state_lock:
        # 前置检查：是否已初始化 + 串口是否正常
        if not gripper_state['initialized']:
            return jsonify({"success": False, "message": "夹爪未初始化，请先调用/gripper/init"}), 400
        if not gripper.serial_port.is_open:
            gripper_state['initialized'] = False  # 重置状态
            return jsonify({"success": False, "message": "夹爪串口已断开，请重新初始化"}), 500

        # 执行释放动作
        try:
            gripper.grap_release()
            gripper_state['last_operation'] = time.time()
            return jsonify({
                "success": True,
                "message": "释放指令已发送",
                "last_operation": gripper_state['last_operation']
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"释放失败: {str(e)}"}), 500


@app.route('/gripper/status', methods=['GET'])
def gripper_status():
    """获取夹爪当前状态"""
    global gripper, gripper_state

    with gripper_state_lock:
        # 主动检查串口状态，避免"假初始化"
        actual_serial_open = gripper.serial_port.is_open
        if gripper_state['initialized'] and not actual_serial_open:
            gripper_state['initialized'] = False  # 修正状态

        return jsonify({
            "initialized": gripper_state['initialized'],
            "serial_port": gripper_state['port'],
            "baud_rate": gripper_state['baud_rate'],
            "serial_open": actual_serial_open,
            "last_operation": gripper_state['last_operation']
        })


@app.route('/gripper/deinit', methods=['POST'])
def gripper_deinit():
    """关闭夹爪串口，重置状态"""
    global gripper, gripper_state

    with gripper_state_lock:
        # 检查是否需要关闭
        if not gripper_state['initialized'] and not gripper.serial_port.is_open:
            return jsonify({"success": False, "message": "夹爪未初始化或串口已关闭"}), 400

        # 执行关闭操作
        try:
            gripper.serial_deinit()
            # 重置夹爪状态
            gripper_state['initialized'] = False
            gripper_state['last_operation'] = None
            return jsonify({"success": True, "message": "夹爪串口已关闭，状态已重置"})
        except Exception as e:
            return jsonify({"success": False, "message": f"关闭失败: {str(e)}"}), 500


# --------------------------- 服务启动 ---------------------------
if __name__ == '__main__':
    # 运行Flask服务（允许外部访问，线程安全模式）
    app.run(host='0.0.0.0', port=11223, debug=False, threaded=True)