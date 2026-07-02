import requests
import time
import json
from typing import Dict, Optional, List, Any
from urllib.parse import urljoin  # 安全拼接URL


class GripperClient:
    """夹爪控制客户端类（与服务端API一一对应）"""

    def __init__(self,
                 server_url: str,
                 name: str = "Gripper",
                 timeout: int = 5,
                 retry_count: int = 2,
                 retry_delay: float = 1.0):
        """
        初始化客户端

        :param server_url: 服务端基础地址（如 http://192.168.1.231:11224/api）
        :param name: 夹爪名称（多设备区分用）
        :param timeout: 请求超时时间（秒）
        :param retry_count: 失败重试次数
        :param retry_delay: 重试间隔（秒）
        """
        # 统一处理URL后缀，避免拼接错误
        self.server_url = server_url.rstrip('/') + '/' if server_url else ''
        self.name = name
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # 记录最后一次操作的响应和时间
        self.last_response: Optional[Dict[str, Any]] = None
        self.last_operation_time: Optional[float] = None

    def _send_request(self, endpoint: str, method: str = "get", data: Optional[Dict] = None) -> Optional[
        Dict[str, Any]]:
        """私有通用请求方法（所有API调用的基础）"""
        url = urljoin(self.server_url, endpoint.lstrip('/'))  # 安全拼接URL
        attempt = 0
        self.last_operation_time = time.time()

        while attempt <= self.retry_count:
            try:
                # 统一请求参数（减少代码重复）
                request_kwargs = {
                    "timeout": self.timeout,
                    "headers": {"Content-Type": "application/json"}
                }

                # 分发请求方法
                if method.lower() == "get":
                    response = requests.get(url, **request_kwargs)
                elif method.lower() == "post":
                    request_kwargs["json"] = data or {}
                    response = requests.post(url, **request_kwargs)
                else:
                    raise ValueError(f"不支持的方法: {method}（仅get/post）")

                # 检查HTTP状态码（4xx/5xx直接抛异常）
                response.raise_for_status()

                # 解析JSON响应（处理空响应场景）
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    raise ValueError("服务端返回非JSON格式数据")

                # 补充元数据（便于排查问题）
                result["request_url"] = url
                result["response_time"] = round(time.time() - self.last_operation_time, 3)

                self.last_response = result
                return result

            except Exception as e:
                attempt += 1
                error_msg = f"{str(e)}（URL: {url}）"

                # 重试逻辑
                if attempt > self.retry_count:
                    print(f"[{self.name}] 请求失败（已重试{self.retry_count}次）: {error_msg}")
                    self.last_response = {
                        "success": False,
                        "message": error_msg,
                        "request_url": url
                    }
                    return None

                print(f"[{self.name}] 请求失败，将重试（{attempt}/{self.retry_count}）: {error_msg}")
                time.sleep(self.retry_delay)

        return None

    def print_response(self, response: Optional[Dict[str, Any]] = None) -> None:
        """格式化打印响应（直观展示结果）"""
        response = response or self.last_response
        if not response:
            print(f"[{self.name}] 无响应数据")
            return

        # 基础信息
        print(f"\n[{self.name}] 响应结果:")
        print(f"  动作: {response.get('action', '未知')}")
        print(f"  状态: {'✅ 成功' if response.get('success') else '❌ 失败'}")
        print(f"  消息: {response.get('message', '无')}")

        # 位置信息（仅当存在时显示）
        if 'position' in response and response['position'] is not None:
            print(f"  位置值: {response['position']}")
            print(f"  位置状态: {response.get('status', '未知')}")

        # 元数据
        if 'request_url' in response:
            print(f"  请求URL: {response['request_url']}")
        if 'response_time' in response:
            print(f"  响应耗时: {response['response_time']} 秒")

    # -------------------------- 客户端核心功能（与服务端API对齐） --------------------------
    def initialize(self) -> bool:
        """初始化夹爪"""
        print(f"[{self.name}] 执行夹爪初始化...")
        result = self._send_request("initialize", "post")
        self.print_response(result)
        return result.get('success', False) if result else False

    def catch(self) -> bool:
        """控制夹爪抓取"""
        print(f"[{self.name}] 执行抓取动作...")
        result = self._send_request("catch", "post")
        self.print_response(result)
        return result.get('success', False) if result else False

    def release(self) -> bool:
        """控制夹爪释放"""
        print(f"[{self.name}] 执行释放动作...")
        result = self._send_request("release", "post")
        self.print_response(result)
        return result.get('success', False) if result else False

    def get_position(self) -> Optional[Dict[str, Any]]:
        """获取夹爪当前位置"""
        print(f"[{self.name}] 读取夹爪位置...")
        result = self._send_request("position", "get")
        self.print_response(result)
        return result

    def get_status(self) -> Optional[Dict[str, Any]]:
        """获取系统（服务端+串口）状态"""
        print(f"[{self.name}] 查询系统状态...")
        result = self._send_request("status", "get")

        if result and result.get('success'):
            print(f"\n[{self.name}] 系统状态:")
            data = result.get('data', {})
            print(f"  串口连接: {'✅ 已连接' if data.get('connected') else '❌ 未连接'}")
            if data.get('connected'):
                print(f"  串口端口: {data.get('port', '未知')}")
                print(f"  波特率: {data.get('baudrate', '未知')}")
            print(f"  服务端时间: {data.get('timestamp', '未知')}")

        return result

    def close(self) -> bool:
        """通知服务端关闭串口"""
        print(f"[{self.name}] 通知服务端关闭串口...")
        result = self._send_request("close", "post")
        self.print_response(result)
        return result.get('success', False) if result else False

    def perform_full_cycle(self, delay: float = 2.0) -> bool:
        """
        完整操作周期：状态检查 → 初始化 → 抓取 → 读位置 → 释放 → 读位置 → 关闭 → 状态检查
        """
        print(f"\n====== 【{self.name}】夹爪完整操作周期开始 ======")
        cycle_success = True

        # 步骤1：初始状态检查

        print(f"\n[步骤1/8] 检查初始系统状态")
        self.get_status()


        # 步骤2：初始化（核心步骤，失败则终止）
        print(f"\n[步骤2/8] 初始化夹爪")
        if not self.initialize():
            print(f"[{self.name}] 初始化失败，终止周期")
            cycle_success = False
            return cycle_success

        # 步骤3：抓取动作
        print(f"\n[步骤3/8] 执行抓取")
        if not self.catch():
            print(f"[{self.name}] 抓取动作失败")
            cycle_success = False

        # 步骤4：抓取后读位置
        print(f"\n[步骤4/8] 抓取后等待{delay}秒并读位置")
        time.sleep(delay)
        self.get_position()

        # 步骤5：释放动作
        print(f"\n[步骤5/8] 执行释放")
        if not self.release():
            print(f"[{self.name}] 释放动作失败")
            cycle_success = False

        # 步骤6：释放后读位置
        print(f"\n[步骤6/8] 释放后等待{delay}秒并读位置")
        time.sleep(delay)
        self.get_position()

        # 步骤7：关闭串口
        print(f"\n[步骤7/8] 关闭串口连接")
        self.close()

        # 步骤8：最终状态检查
        print(f"\n[步骤8/8] 检查最终系统状态")
        self.get_status()

        # 周期总结
        cycle_msg = "✅ 全部成功" if cycle_success else "⚠️ 部分步骤失败"
        print(f"\n====== 【{self.name}】夹爪完整操作周期结束（结果：{cycle_msg}） ======\n")
        return cycle_success


def main():
    """客户端入口：演示单夹爪控制（你的核心使用场景）"""
    # 初始化客户端（服务端地址与port=11224对齐）
    gripper = GripperClient(
        server_url="http://192.168.1.231:11224/api",
        name="MainGripper",  # 夹爪名称（多设备时可修改）
        timeout=8,  # 局域网请求超时可适当延长
        retry_count=2  # 网络不稳定时可增加重试次数
    )

    # 执行完整操作周期
    gripper.perform_full_cycle(delay=2.0)  # delay=2：动作间等待2秒（根据硬件调整）


if __name__ == "__main__":
    # 全局异常捕获（避免程序意外崩溃）
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被手动终止（Ctrl+C）")
    except Exception as e:
        print(f"\n程序运行异常: {str(e)}")
    finally:
        print("程序退出")