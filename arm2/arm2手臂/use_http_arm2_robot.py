import socket
import time
import threading


class Pose:
    """机械臂末端TCP位姿结构体"""
    def __init__(self, x=0.0, y=0.0, z=0.0, r=0.0, p=0.0, _y_=0.0):
        self.x = x  # X坐标
        self.y = y  # Y坐标
        self.z = z  # Z坐标
        self.r = r  # 旋转角度r
        self.p = p  # 旋转角度p
        self._y_ = _y_  # 旋转角度y


class JPose:
    """机械臂关节角度结构体"""

    def __init__(self, j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0):
        self.j1 = j1  # 关节1角度
        self.j2 = j2  # 关节2角度
        self.j3 = j3  # 关节3角度
        self.j4 = j4  # 关节4角度
        self.j5 = j5  # 关节5角度
        self.j6 = j6  # 关节6角度


class RobotController:
    def __init__(self):
        self.count = 0
        self.sock = None
        self.count_lock = threading.Lock()
        # 默认阈值（毫米/度）
        self.default_tolerance = {
            'x': 1.0,
            'y': 1.0,
            'z': 1.0,
            'r': 1.0,
            'p': 1.0,
            '_y_': 1.0,
        }

    def connect_socket(self, ip='127.0.0.1', port=8080, timeout=10):
        """建立socket连接（带超时）"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((ip, port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(None)
            print("Socket connected successfully.")
            return self.sock
        except socket.timeout:
            print(f"Socket connection timeout (>{timeout}s)")
            return None
        except ConnectionRefusedError:
            print("Connection refused: check IP/port")
            return None
        except Exception as e:
            print(f"Socket connection failed: {e}")
            return None

    def split_res(self, res, index, dest):
        """分割响应字符串"""
        tokens = res.split('III')
        if 0 < index <= len(tokens):
            dest[0] = tokens[index - 1]
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
                if len(buf) > 1024 * 1024:
                    print("Response too large (>1MB), truncating")
                    return buf.decode()
            except ConnectionResetError:
                print("Robot disconnected (reset)")
                return None
            except Exception as e:
                print(f"Recv error: {e}")
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
                print(f"Send failed (sent {send_num}/{len(cmd)})")
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
                print(f"Send failed (sent {send_num}/{len(cmd_val)})")
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
                        "current_pose": current,
                        "target_pose": target,
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
        final_errors = {
            'x': abs(final.x - target.x),
            'y': abs(final.y - target.y),
            'z': abs(final.z - target.z),
            'r': abs(final.r - target.r),
            'p': abs(final.p - target.p),
            '_y_': abs(final._y_ - target._y_)
        }
        return {
            "success": False,
            "current_pose": final,
            "target_pose": target,
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
                    print(f"Send failed (sent {send_num}/{len(cmd)})")
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

    def run_joint_point(self, sock, j: JPose, speed):
        def run_point_no_block(sock, j, speed):
            """
            非阻塞方式运行机器人到指定点
            参数:
                sock: 已连接的socket对象
                j: 包含j1-j6关节角度的字典
                speed: 运动速度
            返回:
                0: 成功, -1: 失败
            """
            global count

            # 获取工具姿态
            res, p = get_tool_pose(sock, j)
            if res != 0:
                print("获取关节姿态失败")
                return -1

            try:
                # 构造命令内容
                count += 1
                cmd_val = (f"MoveJ({j['j1']:.3f},{j['j2']:.3f},{j['j3']:.3f},{j['j4']:.3f},{j['j5']:.3f},{j['j6']:.3f},"
                           f"{p['x']:.3f},{p['y']:.3f},{p['z']:.3f},{p['r']:.3f},{p['p']:.3f},{p['_y_']:.3f}, "
                           f"0,0, {speed % 100},100,100,0.000,0.000,0.000,0.000,300,0,0,0,0,0,0,0)")

                # 构造完整命令
                cmd = f"/f/bIII{count % 100 + 100}III201III{len(cmd_val)}III{cmd_val}III/b/f"

                # 发送命令
                send_num = sock.send(cmd.encode('utf-8'))
                if send_num != len(cmd):
                    print(f"socket发送错误，行号: {locals()['__line__']}")
                    return -1

                # 接收响应
                buf = b''
                while True:
                    try:
                        data = sock.recv(256)
                        if not data:
                            print(f"socket读取错误，行号: {locals()['__line__']}")
                            return -1
                        buf += data
                        if len(buf) > 0:
                            break
                    except BlockingIOError:
                        # 非阻塞模式下没有数据可用时继续等待
                        continue
                    except Exception as e:
                        print(f"接收数据错误: {e}")
                        return -1

                # 解析响应
                response = buf.decode('utf-8')
                valid = split_res(response, 3, "")
                try:
                    res_num = int(valid)
                    if res_num != 201:
                        print(f"运行点错误: {response}")
                        return -1
                except ValueError:
                    print(f"响应解析错误: {response}")
                    return -1

                return 0

            except Exception as e:
                print(f"执行过程中发生错误: {e}")
                return -1

    def run_point_with_safety(self, sock, p: Pose, speed, tolerance=None, timeout=15):
        """带到位判断的PTP运动（弧线）"""
        # 获取原始位姿
        original_pose = Pose()
        if self.get_tcp_pose(sock, original_pose) != 0:
            return {"success": False, "message": "获取初始位姿失败"}

        # 计算目标偏移量（目标位姿 - 原始位姿）
        target_offset = (
            p.x - original_pose.x,
            p.y - original_pose.y,
            p.z - original_pose.z,
            p.r - original_pose.r,
            p.p - original_pose.p,
            p._y_ - original_pose._y_
        )

        # 执行移动命令
        move_start = time.time()
        move_res = self.run_point(sock, p, speed)
        if move_res != 0:
            return {
                "success": False,
                "message": "PTP移动命令被机械臂拒绝",
                "original_pose": original_pose,
                "target_pose": p
            }

        # 等待到位验证
        verification = self.wait_for_move_complete(
            sock,
            target_offset=target_offset,
            original_pose=original_pose,
            tolerance=tolerance,
            timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": original_pose,
            "target_pose": p
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
                    print(f"Send failed (sent {send_num}/{len(cmd)})")
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
        target_offset = (
            c_p.x - original_pose.x,
            c_p.y - original_pose.y,
            c_p.z - original_pose.z,
            c_p.r - original_pose.r,
            c_p.p - original_pose.p,
            c_p._y_ - original_pose._y_
        )

        # 执行移动命令
        move_start = time.time()
        move_res = self.run_point_line(sock, c_p, speed)
        if move_res != 0:
            return {
                "success": False,
                "message": "直线移动命令被机械臂拒绝",
                "original_pose": original_pose,
                "target_pose": c_p
            }

        # 等待到位验证
        verification = self.wait_for_move_complete(
            sock,
            target_offset=target_offset,
            original_pose=original_pose,
            tolerance=tolerance,
            timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": original_pose,
            "target_pose": c_p
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
                "original_pose": original
            }

        verification = self.wait_for_move_complete(
            sock,
            target_offset=(x, y, z, r, p, _y_),
            original_pose=original,
            tolerance=tolerance,
            timeout=timeout
        )

        return {
            "move_accepted": move_res == 0,
            "verification": verification,
            "total_time": round(time.time() - move_start, 2),
            "original_pose": original
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
                    print(f"Send failed (sent {send_num}/{len(cmd)})")
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
            print("Socket closed.")


if __name__ == "__main__":
    print("Start")
    robot = RobotController()
    sock = robot.connect_socket(ip='192.168.1.210', port=8080)
    if not sock:
        exit(1)
    #
    print("===  机器人2控制 ===")
    ptp_target = Pose(x=135.429, y=-150.135, z=1073.908, r=-89.964, p=-2.869, _y_=-137.097)
    # 使用默认阈值（x/y/z:±1mm）
    ptp_res = robot.run_point_with_safety(
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

    # # 1.1 测试带到位判断的PTP运动（弧线）
    # print("\n=== 测试1：PTP运动（弧线）带到位判断 ===")
    # # 定义目标位姿（示例值，需根据实际场景修改）
    # ptp_target = Pose(x=315.109, y=306.038, z=607.385, r=180.0, p=0, _y_=-50)
    # # 使用默认阈值（x/y/z:±1mm）
    # ptp_res = robot.run_point_with_safety(
    #     sock,
    #     p=ptp_target,
    #     speed=50,
    #     tolerance={'x': 1, 'y': 1, 'z': 1, }  # 自定义x/y轴阈值
    # )
    # print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    # print(f"目标位置：x={ptp_res['target_pose'].x:.2f}, y={ptp_res['target_pose'].y:.2f}, z={ptp_res['target_pose'].z}")
    # print(
    #     f"实际位置：x={ptp_res['verification']['current_pose'].x:.2f}, y={ptp_res['verification']['current_pose'].y:.2f}, z={ptp_res['verification']['current_pose'].z:.2f}")
    # print(
    #     f"误差：x={ptp_res['verification']['errors']['x']:.2f}mm, y={ptp_res['verification']['errors']['y']:.2f}mm, z={ptp_res['verification']['errors']['z']:.2f}mm")
    # print(f"耗时：{ptp_res['total_time']}秒，信息：{ptp_res['verification']['message']}")
    #
    #
    # # 1.2 测试带到位判断的PTP运动（弧线）
    # print("\n=== 测试1：PTP运动（弧线）带到位判断 ===")
    # # 定义目标位姿（示例值，需根据实际场景修改）
    # ptp_target = Pose(x=135.429, y=-150.135, z=1073.908, r=-89.964, p=-2.869, _y_=-137.097)
    # # 使用默认阈值（x/y/z:±1mm）
    # ptp_res = robot.run_point_with_safety(
    #     sock,
    #     p=ptp_target,
    #     speed=50,
    #     tolerance={'x': 1.5, 'y': 1.5}  # 自定义x/y轴阈值
    # )
    # print(f"PTP移动结果：{'成功' if ptp_res['verification']['success'] else '失败'}")
    # print(f"目标位置：x={ptp_res['target_pose'].x:.2f}, y={ptp_res['target_pose'].y:.2f}, z={ptp_res['target_pose'].z}")
    # print(
    #     f"实际位置：x={ptp_res['verification']['current_pose'].x:.2f}, y={ptp_res['verification']['current_pose'].y:.2f}, z={ptp_res['verification']['current_pose'].z:.2f}")
    # print(
    #     f"误差：x={ptp_res['verification']['errors']['x']:.2f}mm, y={ptp_res['verification']['errors']['y']:.2f}mm, z={ptp_res['verification']['errors']['z']:.2f}mm")
    # print(f"耗时：{ptp_res['total_time']}秒，信息：{ptp_res['verification']['message']}")


    # # 2. 测试带到位判断的直线运动（不使用）
    # print("\n=== 测试2：直线运动带到位判断 ===")
    # # 定义目标位姿（示例值）
    # line_target = Pose(x=-129.0, y=-12.0, z=915.0, r=-89.0, p=-1.0, _y_=-135.0)
    # line_res = robot.run_point_line_with_safety(
    #     sock,
    #     c_p=line_target,
    #     speed=20,
    #     tolerance={'z': 2.0}  # 放宽z轴阈值
    # )
    # print(f"直线移动结果：{'成功' if line_res['verification']['success'] else '失败'}")
    # print(f"目标位置：x={line_res['target_pose'].x:.2f}, z={line_res['target_pose'].z:.2f}")
    # print(f"实际位置：x={line_res['verification']['current_pose'].x:.2f}, z={line_res['verification']['current_pose'].z:.2f}")
    # print(f"误差：x={line_res['verification']['errors']['x']:.2f}mm, z={line_res['verification']['errors']['z']:.2f}mm")
    # print(f"耗时：{line_res['total_time']}秒，信息：{line_res['verification']['message']}")

    robot.close_socket()
