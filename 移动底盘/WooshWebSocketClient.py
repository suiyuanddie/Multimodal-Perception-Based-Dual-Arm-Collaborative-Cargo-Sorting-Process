import websocket
import json
import rel
import threading
import time

kTypeUndefined = 0  # 未定义的
kNav = 1  # 导航
kStepCtrl = 2  # 单步控制
kSecondposEnter = 3  # 二次定位进入
kSecondposQuit = 4  # 二次定位退出
kCarry = 5  # 搬运动作
kWait = 6  # 等待
kCharge = 7  # 充电


class WooshWebSocketClient:
    def __init__(self, url, debug=False):
        self.subscription_events = {}
        self.subscription_data = {}
        self.subscription_lock = threading.Lock()
        self.thread = None
        self.url = url
        self.ws_request = None
        self.ws_subscribe = None
        self.subscriptions = {}
        self.debug = debug
        websocket.enableTrace(self.debug)

    @staticmethod
    def on_open(ws):
        print("WebSocket connection opened")

    def on_message(self, ws, message):
        message = json.loads(message)
        sn_num = message.get("sn")
        if sn_num is not None:
            with self.subscription_lock:
                self.subscription_data[sn_num] = message
                self.subscription_events[sn_num].set()
        else:
            self.exec_callback(message["type"], message)

    def exec_callback(self, message_type, message):
        """执行回调函数
        """
        with self.subscription_lock:
            if message_type in self.subscriptions:
                try:
                    self.subscriptions[message_type](message)
                except Exception as e:
                    print(f"Error executing callback for {message_type}: {e}")

    @staticmethod
    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    @staticmethod
    def on_close(ws):
        print("WebSocket connection closed")

    def connect(self):
        self.close()
        self.ws_subscribe = websocket.WebSocketApp(self.url,
                                                   on_open=self.on_open,
                                                   on_message=self.on_message,
                                                   on_error=self.on_error,
                                                   on_close=self.on_close)

        self.ws_request = websocket.WebSocketApp(self.url,
                                                 on_open=self.on_open,
                                                 on_message=self.on_message,
                                                 on_error=self.on_error,
                                                 on_close=self.on_close)

        # 后台 rel 调度执行
        self.ws_subscribe.run_forever(dispatcher=rel, reconnect=10)
        self.ws_request.run_forever(dispatcher=rel, reconnect=10)
        self.thread = threading.Thread(target=rel.dispatch)
        self.thread.start()

    def close(self):
        if self.ws_subscribe:
            self.ws_subscribe.close()
        if self.ws_request:
            self.ws_request.close()
        if self.thread:
            rel.abort()
            self.thread.join()

    def add_topic_callback(self, message_type: str, cb_func):
        """添加话题回调函数
        """
        with self.subscription_lock:
            self.subscriptions[message_type] = cb_func

    def submit_subscriptions(self):
        """发送订阅请求
        """
        msg = {
            "type": "woosh.Subscription",
            "body": {
                "sub": True,
                "topics": list(self.subscriptions.keys())
            }
        }
        self.ws_subscribe.send(json.dumps(msg))

    def remove_topic_callback(self, message_type):
        """移除话题回调函数
        """
        with self.subscription_lock:
            if message_type in self.subscriptions:
                del self.subscriptions[message_type]

    def request(self, message_type, body=None, timeout=8, send_type='request'):
        """发送请求，根据 sn 序列号保证 req 和 resp 匹配
        """
        sn_num = int(time.time())

        subscription_event = threading.Event()#通过序列号（sn）+ 线程事件（threading.Event） 机制，解决了异步通信中 “请求与响应如何对应” 的问题：每个请求生成唯一 sn，并关联一个 Event 对象；收到响应时，根据 sn 找到对应的 Event 并触发，确保请求线程能准确获取自己的响应；

        with self.subscription_lock:
            self.subscription_events[sn_num] = subscription_event

        if body is None or type(body) is not dict:
            msg = {"type": message_type, "sn": sn_num}
        else:
            msg = {"type": message_type, "sn": sn_num, "body": body}
        if send_type == 'request':
            self.ws_request.send(json.dumps(msg))
        if send_type == 'subscribe':
            self.ws_subscribe.send(json.dumps(msg))

        if subscription_event.wait(timeout):
            with self.subscription_lock:
                subscription_data = self.subscription_data[sn_num]
                self.subscription_data[sn_num] = None
                del self.subscription_events[sn_num]
                # print("subscription_data", subscription_data)
                if subscription_data.get('ok'):
                    return subscription_data.get('body')
                else:
                    print(
                        f"Fail to get ok for response to {message_type} sn: {sn_num}")
                    return None
        else:
            print(
                f"Timeout waiting for response to {message_type} sn: {sn_num}")
            return None

    def request_try(self, message_type, body=None, send_type='request', timeout=8):
        ret = False
        try:
            ret = self.request(message_type, body=body, timeout=timeout, send_type=send_type)
        except Exception as e:
            print(f"Error requesting {message_type}: {e}, body: {body}")
        finally:
            return ret


class WooshApi(WooshWebSocketClient):
    def __init__(self, url, debug=False):
        super().__init__(url, debug)
        self.connect()

    # =================================================
    # 机器人信息相关
    # =================================================
    def robot_info(self):
        return self.request_try("woosh.robot.RobotInfo")

    def robot_general(self):
        return self.request_try("woosh.robot.General")

    def robot_setting(self):
        return self.request_try("woosh.robot.Setting")

    def robot_state(self):
        return self.request_try("woosh.robot.RobotState")

    def robot_mode(self):
        return self.request_try("woosh.robot.Mode")

    def robot_pose_speed(self):
        return self.request_try("woosh.robot.PoseSpeed")

    def robot_battery(self, send_type='request'):
        return self.request_try(message_type="woosh.robot.Battery", send_type=send_type)

    def robot_network(self):
        return self.request_try("woosh.robot.Network")

    def robot_scene(self):
        return self.request_try("woosh.robot.Scene")

    def robot_task_proc(self):
        return self.request_try("woosh.robot.TaskProc")

    def robot_device_state(self):
        return self.request_try("woosh.robot.DeviceState")

    def robot_hardware_state(self):
        return self.request_try("woosh.robot.HardwareState")


    def robot_operation_state(self):
        return self.request_try("woosh.robot.OperationState")

    def robot_model(self):
        return self.request_try("woosh.robot.Model")

    def robot_task_history(self):
        return self.request_try("woosh.robot.TaskHistory")

    def robot_status_code(self, robot_id):
        body = {"robotId": robot_id}
        return self.request_try("woosh.robot.count.StatusCodes", body)

    def robot_abnormal_codes(self):
        return self.request_try("woosh.robot.count.AbnormalCodes")

    def robot_is_impede(self):
        """判断机器人是否行驶路线被阻碍
        kNavBitUndefined	0	未定义
        kNarrow	 1	狭窄通道
        kGuide	 2	引导到达
        kInaLift 4  乘梯中
        kImpede	 8	阻碍
        kQRCode	 16 二维码
        """
        msg = self.robot_operation_state()
        if msg is None:
            return None
        return msg.get('nav') & 0b1000 == 8 if msg.get('nav') else None

    # =================================================
    # 机器人配置相关
    # =================================================
    def setting_identity(self, name, robot_id=None):
        body = {"name": name}
        if robot_id is not None:
            body["robot_id"] = robot_id
        return self.request_try("woosh.robot.setting.Identity", body)

    def setting_auto_charge(self, allow=True):
        body = {"allow": allow}
        return self.request_try("woosh.robot.setting.AutoCharge", body)

    def setting_auto_park(self, allow=True):
        body = {"allow": allow}
        return self.request_try("woosh.robot.setting.AutoPark", body)

    def setting_power(self, alarm=10, low=20, idle=80, full=98):
        body = {"alarm": alarm, "low": low, "idle": idle, "full": full}
        return self.request_try("woosh.robot.setting.Power", body)

    # =================================================
    # 机器人场景地图相关
    # =================================================
    def map_scene_list(self):
        """获取场景列表"""
        return self.request_try("woosh.map.SceneList")

    def map_scene_data(self, scene_name):
        """获取场景数据"""
        body = {"name": scene_name}
        return self.request_try("woosh.map.SceneData", body)

    def map_download(self, scene_name):
        return self.request_try("woosh.map.Download", body={"sceneName": scene_name})

    def map_upload(self):
        return self.request_try("woosh.map.Upload")

    # =================================================
    # 机器人请求相关
    # =================================================
    def robot_switch_work_mode(self, mode: int):
        """切换工作模式
        kWorkModeUndefined	0	未定义的
        kDeployMode	        1	部署模式
        kTaskMode           2	任务模式
        kScheduleMode       3   调度模式
        """
        return self.request_try("woosh.robot.SwitchWorkMode", {"mode": mode})

    def robot_init_robot(self, body):
        if body.get('pose') is None:
            return self.request_try("woosh.robot.InitRobot", {"isRecord": True})
        else:
            return self.request_try("woosh.robot.InitRobot", body)

    def robot_set_robot_pose(self, x, y, theta):
        return self.request_try("woosh.robot.SetRobotPose",
                                {"pose": {"x": x, "y": y, "theta": theta}})

    def robot_switch_control_mode(self, mode: int):
        """切换控制模式
        kControlModeUndefined	0	未定义的
        kControlModeManual	    1	手动模式
        kControlModeAuto	    2	自动模式
        """
        return self.request_try("woosh.robot.SwitchControlMode", {"mode": mode})

    def robot_switch_map(self, scene_name, map_name):
        return self.request_try("woosh.robot.SwitchMap",
                                {"sceneName": scene_name, "mapName": map_name})

    def robot_speak(self, text):
        return self.request_try("woosh.robot.Speak", {"text": text})

    def robot_exec_task(self, target: dict, poses: list):
        """执行任务
        target: {"x": 0, "y": 0, "theta": 0}
        path: [{"x": 0, "y": 0, "theta": 0}, ...]
        """
        task_id = int(time.time())
        body = {
            "taskId": task_id,
            "type": kNav,
            "planPath": {
                "planPath": [
                    {
                        "path": {
                            "poses": poses
                        },
                        "target": target
                    }
                ]
            }
        }
        return self.request_try("woosh.robot.ExecTask", body)
    
    def robot_move_poses(self, poses: list, task_id = int(time.time()), mapid=0, destmapid=0):
        body = {
            "taskId": task_id,
            "type": kNav,
            "direction": 0,
            "taskTypeNo":0,
            "planPath": {
                "planPath": [
                    {
                        "path": {
                            "poses": poses
                        },
                        "target": poses[-1],
                        "optimal":9,
                        "map_id": mapid,                        
                        "destMapId": destmapid
                    }
                ]
            }
        }
        return self.request_try("woosh.robot.ExecTask", body)

    def robot_charge_task(self, mark_no=None):
        task_id = int(time.time())
        body = {
            "taskId": task_id,
            "type": 3,
        }
        if mark_no:
            body["markNo"] = mark_no
        return self.request_try("woosh.robot.ExecTask", body)

    def robot_go_to(self, x, y, theta):
        target = {"x": x, "y": y, "theta": theta}
        poses = [{"x": x, "y": y, "theta": theta}]
        return self.robot_exec_task(target, poses)
    # def robot_go_to(self, x, y, theta):
    #     target = {"x": x, "y": y, "theta": theta}
    #     poses = [{"x": x, "y": y, "theta": theta}]
    #     return self.robot_exec_task(target, poses)
    
    def robot_go_to_poses(self, poses: list,):
        return self.robot_move_poses(poses)


    def robot_go_to_markno(self, markno, task_id= int(time.time()), tasktypeno=3, kNav=1):
        task_id = int(time.time())
        body = {
            "taskId": task_id,
            "type": kNav,
            "taskTypeNo":tasktypeno,
            "markNo": markno
        }

        return self.request_try("woosh.robot.ExecTask", body)
    
    def robot_cancel_task(self):
        return self.robot_action_order(order=4)

    # 控制任务的状态，可以用于暂停任务、开始任务、继续任务、取消任务等
    def robot_action_order(self, order=0):
        """
        kOrderUndefined	0	未定义的
        kStart	1	开始(弃用)
        kPause	2	暂停
        kContinue	3	继续
        kCancel	4	取消
        kRecover	5	恢复(单机任务有效)
        kWaitBreak	6	等待打断
        kTmCtrl	7	交通管制
        kReleaseCtrl	8	解除管制
        """
        return self.request_try("woosh.robot.ActionOrder", {"order": order})

    # 规划路径，发送起点和终点以及绕行的上线
    def robot_plan_nav_path(self, start: dict, end: dict, tolerance=0):
        """规划导航路径
        start: {"x": 0, "y": 0, "theta": 0}
        end: {"x": 0, "y": 0, "theta": 0}
        tolerance: 允许绕行范围, 单位为米
        """
        body = {
            "start": start,
            "end": end,
            "tolerance": tolerance
        }
        return self.request_try("woosh.robot.PlanNavPath", body)
    
    # 设置机器人的速度，包括线速度(移动的速度)、角速度(旋转的速度)
    def robot_twist(self, linear, angular):
        """机器人运动
        linear: 线速度, 单位为米/秒
        angular: 角速度, 单位为弧度/秒
        """
        body = {
            "linear": linear,
            "angular": angular
        }
        return self.request_try("woosh.robot.Twist", body)

    # 设置机器人是否被占用，bool类型(true：被占用，false：没被占用)
    def robot_occupancy(self, occupy: bool):
        body = {
            "occupy": occupy
        }
        return self.request_try("woosh.robot.SetOccupancy", body)

    # 暂不清楚
    def robot_open_hotspot(self, order_type):
        body = {
            "order": 5,
            "enable": order_type
        }
        return self.request("woosh.robot.RobotWiFi", body)
    
    # 切换模糊到达
    # 1：模糊到达
    # 2: 精准到达
    def robot_change_nav_mod(self, nav_type):
        body = {
            "nav_mode": {
                "type": nav_type
            }
        }
        return self.request_try("woosh.robot.ChangeNavMode", body)
    

