import numpy as np
import cv2
import time
import pyrealsense2 as rs
import os
from datetime import datetime
from flask import Flask, Response, jsonify
import threading
import base64

class IntelD435:
    def __init__(self, width=640, height=480, fps=15, show_window=False):
        # 初始化状态标记
        self.initialized = False
        self.running = False
        self.last_frame_time = 0
        self.is_processing = False  # 避免缓存处理并发冲突
        
        # 图像存储路径（核心：自动保存到output的固定文件名文件）
        self.output_root = "output/intelD435"
        self.output_color_path = f"{self.output_root}/color/color_image.png"  # 自动保存的彩色图（固定名）
        self.output_depth_path = f"{self.output_root}/depth/depth_filtered_image.png"  # 自动保存的滤波深度图（固定名）
        self.output_depth_vis_path = f"{self.output_root}/depth_vis/depth_vis_image.png"  # 自动保存的深度可视化图（固定名）
        self.temp_dir = f"{self.output_root}/image_temp"  # 临时目录（保留但接口不再使用）
        # 接口触发保存路径（基于output的固定文件）
        self.req_save_root = f"{self.output_root}/request_saved"
        self.req_save_color = f"{self.req_save_root}/color"
        self.req_save_depth = f"{self.req_save_root}/depth"
        self.req_save_depth_vis = f"{self.req_save_root}/depth_vis"
        
        # 确保所有路径存在
        all_dirs = [
            f"{self.output_root}/color", f"{self.output_root}/depth", f"{self.output_root}/depth_vis",
            self.temp_dir, self.req_save_color, self.req_save_depth, self.req_save_depth_vis
        ]
        for dir_path in all_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # 临时文件路径（保留实时更新，但接口不再读取）
        self.temp_color_path = os.path.join(self.temp_dir, 'latest_color.png')
        self.temp_depth_path = os.path.join(self.temp_dir, 'latest_depth_16bit.png')
        self.temp_depth_vis_path = os.path.join(self.temp_dir, 'latest_depth_vis.png')
        
        # 帧缓存配置（原有逻辑：每10帧处理一次，更新output的固定文件）
        self.frame_cache = []          # 缓存格式：[(color1, depth1, vis1), ...]
        self.cache_capacity = 10       # 固定缓存10组帧
        
        # 相机核心对象
        self.pipeline = None
        self.config = None
        self.align = None
        self.colorizer = None
        
        try:
            # 1. 配置相机流（原有逻辑）
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            if not device:
                raise RuntimeError("未检测到Intel D435设备")
            
            # 启用彩色流（8位BGR）和深度流（16位Z16）
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            # 2. 启动相机管道
            print("正在启动相机...")
            self.profile = self.pipeline.start(self.config)
            print("相机管道启动成功")
            
            # 3. 初始化对齐和颜色映射器
            self.align = rs.align(rs.stream.color)
            self.colorizer = rs.colorizer()
            
            # 4. 获取相机内参
            self.get_intrinsics()
            self.print_intrinsics()
            
            # 5. 标记就绪并启动采集线程
            self.initialized = True
            self.running = True
            print(f"\n相机初始化完成！参数：{width}x{height} @ {fps}FPS")
            print(f"=== 保存逻辑说明 ===")
            print(f"1. 自动保存：每10帧处理后更新 output/intelD435/ 下的固定文件名文件：")
            print(f"   - 彩色图：{self.output_color_path}")
            print(f"   - 滤波深度图：{self.output_depth_path}")
            print(f"   - 深度可视化图：{self.output_depth_vis_path}")
            print(f"2. 接口触发保存：调用 /request_images 时，从output目录复制文件到 request_saved/...")
            
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            print("相机采集线程已启动（每10帧更新output固定文件+临时文件实时更新）")
            
        except Exception as e:
            print(f"\n相机初始化失败！原因：{str(e)}")
            print("排查建议：1. USB3.0连接 2. 驱动正常 3. 分辨率/FPS未超硬件支持")
            self.cleanup()
    
    def get_intrinsics(self):
        """获取相机内参（原有逻辑）"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("获取初始帧失败，无法提取内参")
            
            self.color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            
        except Exception as e:
            print(f"获取内参失败：{str(e)}")
            self.color_intrinsics = None
            self.depth_intrinsics = None
    
    def print_intrinsics(self):
        """打印内参（原有逻辑）"""
        if not self.color_intrinsics or not self.depth_intrinsics:
            print("内参未获取，跳过打印")
            return
        
        print("\n=== 彩色相机内参 ===")
        print(f"  分辨率: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
        print(f"  焦距: fx={self.color_intrinsics.fx:.2f}, fy={self.color_intrinsics.fy:.2f}")
        print(f"  主点: cx={self.color_intrinsics.ppx:.2f}, cy={self.color_intrinsics.ppy:.2f}")
        
        print("\n=== 深度相机内参 ===")
        print(f"  分辨率: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
        print(f"  焦距: fx={self.depth_intrinsics.fx:.2f}, fy={self.depth_intrinsics.fy:.2f}")
        print(f"  主点: cx={self.depth_intrinsics.ppx:.2f}, cy={self.depth_intrinsics.ppy:.2f}")
    
    def get_frames(self):
        """获取对齐后的帧（原有逻辑）"""
        try:
            if not self.pipeline or not self.running:
                return None, None, None
            
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("警告：获取到无效帧，跳过本次采集")
                return None, None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_vis_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            
            return color_image, depth_image, depth_vis_image
        
        except Exception as e:
            print(f"获取帧失败：{str(e)}")
            return None, None, None
    
    def smooth_depth_frames(self, depth_frames):
        """深度图中位数滤波（原有逻辑，处理All-NaN）"""
        depth_stack = np.stack(depth_frames, axis=0).astype(np.float32)
        depth_stack[depth_stack == 0] = np.nan
        
        valid_mask = ~np.isnan(depth_stack)
        valid_count = np.sum(valid_mask, axis=0)
        filtered_depth = np.zeros_like(depth_stack[0], dtype=np.float32)
        
        if np.any(valid_count > 0):
            with np.errstate(invalid='ignore'):
                filtered_depth[valid_count > 0] = np.nanmedian(
                    depth_stack[:, valid_count > 0], 
                    axis=0
                )
        
        filtered_depth[np.isnan(filtered_depth)] = 0
        return filtered_depth.astype(np.uint16)
    
    def save_processed_frames(self, latest_color, filtered_depth, latest_vis):
        """修改：每10帧处理后更新output目录的固定文件名文件（覆盖式）"""
        # 保存彩色图（覆盖更新固定文件）
        cv2.imwrite(self.output_color_path, latest_color)
        # 保存滤波后深度图（覆盖更新固定文件）
        cv2.imwrite(self.output_depth_path, filtered_depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # 保存深度可视化图（覆盖更新固定文件）
        cv2.imwrite(self.output_depth_vis_path, latest_vis)
        # print(f"✅ 自动更新output固定文件完成（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）")
    
    def save_request_frames(self):
        """修改：接口调用时，从output目录复制固定文件到request_saved目录（而非读取临时文件）"""
        # 检查output目录的固定文件是否存在
        output_files = [self.output_color_path, self.output_depth_path, self.output_depth_vis_path]
        if not all(os.path.exists(f) for f in output_files):
            print("警告：output目录的固定文件不完整，接口触发保存失败")
            return False
        
        # 生成带接口调用标识的时间戳（避免重名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        req_prefix = "request_"  # 前缀标识接口触发的保存
        
        try:
            # 从output复制彩色图到request_saved
            color_img = cv2.imread(self.output_color_path)
            color_save = os.path.join(self.req_save_color, f"{req_prefix}color_{timestamp}.png")
            cv2.imwrite(color_save, color_img)
            
            # 从output复制滤波深度图到request_saved（16位）
            depth_img = cv2.imread(self.output_depth_path, cv2.IMREAD_UNCHANGED)
            depth_save = os.path.join(self.req_save_depth, f"{req_prefix}depth_filtered_{timestamp}.png")
            cv2.imwrite(depth_save, depth_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
            # 从output复制深度可视化图到request_saved
            vis_img = cv2.imread(self.output_depth_vis_path)
            vis_save = os.path.join(self.req_save_depth_vis, f"{req_prefix}depth_vis_{timestamp}.png")
            cv2.imwrite(vis_save, vis_img)
            
            print(f"✅ 接口调用保存完成：{color_save.split('/')[-1]}（从output目录复制）")
            return True
        
        except Exception as e:
            print(f"❌ 接口触发保存失败：{str(e)}")
            return False
    
    def capture_loop(self):
        """核心采集循环（原有逻辑：实时更新临时文件+每10帧更新output固定文件）"""
        while self.running:
            start_time = time.time()
            
            color_img, depth_img, vis_img = self.get_frames()
            if color_img is not None and depth_img is not None and vis_img is not None:
                # 1. 实时更新临时文件（保留，但接口不再使用）
                cv2.imwrite(self.temp_color_path, color_img)
                cv2.imwrite(self.temp_depth_path, depth_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                cv2.imwrite(self.temp_depth_vis_path, vis_img)
                self.last_frame_time = time.time()
                
                # 2. 缓存帧并每10帧更新output固定文件
                if not self.is_processing and len(self.frame_cache) < self.cache_capacity:
                    self.frame_cache.append((color_img.copy(), depth_img.copy(), vis_img.copy()))
                
                if not self.is_processing and len(self.frame_cache) >= self.cache_capacity:
                    self.is_processing = True
                    latest_color = self.frame_cache[-1][0]  # 10帧中最新一帧彩色图
                    all_depth = [frame[1] for frame in self.frame_cache]  # 10帧深度图用于滤波
                    latest_vis = self.frame_cache[-1][2]  # 对应彩色图的深度可视化图
                    
                    filtered_depth = self.smooth_depth_frames(all_depth)
                    self.save_processed_frames(latest_color, filtered_depth, latest_vis)  # 更新output固定文件
                    
                    self.frame_cache.clear()
                    self.is_processing = False
            
            # 控制帧率（15FPS）
            elapsed = time.time() - start_time
            time.sleep(max(0.001, 1/15 - elapsed))
    
    def cleanup(self):
        """资源清理（原有逻辑）"""
        self.running = False
        self.is_processing = False
        self.frame_cache.clear()
        
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("相机管道已安全关闭")
            except Exception as e:
                print(f"关闭相机管道时出错：{str(e)}")
        
        self.pipeline = None
        self.initialized = False
    
    def stop(self):
        """外部调用：停止相机"""
        self.cleanup()


# ---------------------- 全局变量与Flask服务 ----------------------
app = Flask(__name__)
camera = None  # 全局相机实例


def init_camera_safe():
    """安全初始化相机（原有逻辑）"""
    global camera
    if camera and camera.initialized and camera.running:
        return True
    
    for retry in range(2):
        print(f"正在初始化相机（第{retry+1}次尝试）...")
        camera = IntelD435(width=640, height=480, fps=15, show_window=False)
        if camera.initialized and camera.running:
            return True
        time.sleep(1)
    
    return False


def run_flask_server():
    """Flask服务线程（原有逻辑）"""
    global app
    print(f"\nFlask服务启动！监听地址：http://0.0.0.0:11228")
    app.run(
        host='0.0.0.0',
        port=11228,
        debug=False,
        use_reloader=False,
        threaded=True
    )


@app.route('/')
def index():
    """首页：更新接口返回逻辑说明（明确基于output固定文件）"""
    return """
    <h1>Intel D435 相机服务 API 指南（接口返回output固定文件）</h1>
    <p><strong>服务状态</strong>：<a href="/status" target="_blank">查看相机与文件状态</a></p>
    
    <h3>核心功能说明</h3>
    <ul>
        <li><strong>自动更新</strong>：相机每缓存10组帧，执行深度滤波后覆盖更新 output/intelD435/ 下的固定文件：
            <ul>
                <li>彩色图：output/intelD435/color/color_image.png</li>
                <li>滤波深度图：output/intelD435/depth/depth_filtered_image.png</li>
                <li>深度可视化图：output/intelD435/depth_vis/depth_vis_image.png</li>
            </ul>
        </li>
        <li><strong>接口返回</strong>：调用 /request_images 时，返回上述output固定文件的Base64数据（非临时文件）</li>
        <li><strong>接口保存</strong>：调用接口时同步从output复制文件到 request_saved 目录，便于追溯</li>
    </ul>
    
    <h3>核心接口</h3>
    <p>接口：<code>GET /request_images</code> 或 <code>POST /request_images</code></p>
    <p>功能：1. 返回output目录固定文件的Base64编码；2. 同步复制文件到 request_saved 目录</p>
    <p>示例请求：<a href="/request_images" target="_blank">点击获取并保存output文件</a></p>
    
    <h3>辅助预览接口</h3>
    <ul>
        <li><a href="/color" target="_blank">/color</a> → 预览 output/color/color_image.png</li>
        <li><a href="/depth" target="_blank">/depth</a> → 下载 output/depth/depth_filtered_image.png（16位滤波深度）</li>
        <li><a href="/depth_vis" target="_blank">/depth_vis</a> → 预览 output/depth_vis/depth_vis_image.png</li>
        <li><a href="/restart" target="_blank">/restart</a> → 重启相机（故障恢复）</li>
    </ul>
    
    <h3>文件路径汇总</h3>
    <table border="1" cellpadding="6">
        <tr>
            <th>文件类型</th>
            <th>自动更新路径（固定名）</th>
            <th>接口保存路径（带时间戳）</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>彩色图</td>
            <td>output/intelD435/color/color_image.png</td>
            <td>output/intelD435/request_saved/color/request_color_xxxx.png</td>
            <td>10帧中最新一帧</td>
        </tr>
        <tr>
            <td>滤波深度图</td>
            <td>output/intelD435/depth/depth_filtered_image.png</td>
            <td>output/intelD435/request_saved/depth/request_depth_filtered_xxxx.png</td>
            <td>10帧中位数滤波后结果</td>
        </tr>
        <tr>
            <td>深度可视化图</td>
            <td>output/intelD435/depth_vis/depth_vis_image.png</td>
            <td>output/intelD435/request_saved/depth_vis/request_depth_vis_xxxx.png</td>
            <td>与彩色图同步</td>
        </tr>
    </table>
    """


@app.route('/status')
def get_status():
    """返回服务状态（新增output文件存在性检查）"""
    global camera
    status = {
        "service_running": True,
        "camera_connected": False,
        "camera_initialized": False,
        "camera_running": False,
        "frame_cache_progress": "0/10",
        "last_frame_timestamp": None,
        "output_files_status": {  # 新增：output目录固定文件状态
            "color_exists": False,
            "depth_exists": False,
            "vis_exists": False,
            "last_update_time": None,
            "note": "所有文件存在时，接口才能正常返回数据"
        },
        "error_message": ""
    }
    
    if camera:
        status["camera_initialized"] = camera.initialized
        status["camera_running"] = camera.running
        status["camera_connected"] = camera.initialized and camera.running
        status["frame_cache_progress"] = f"{len(camera.frame_cache)}/{camera.cache_capacity}"
        
        # 检查output目录固定文件状态
        output_files = [camera.output_color_path, camera.output_depth_path, camera.output_depth_vis_path]
        status["output_files_status"]["color_exists"] = os.path.exists(output_files[0])
        status["output_files_status"]["depth_exists"] = os.path.exists(output_files[1])
        status["output_files_status"]["vis_exists"] = os.path.exists(output_files[2])
        
        # 获取output文件最后更新时间（取最新修改的文件时间）
        if all(status["output_files_status"].values()):
            max_mtime = max(os.path.getmtime(f) for f in output_files)
            status["output_files_status"]["last_update_time"] = datetime.fromtimestamp(max_mtime).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 最后一帧采集时间
        if camera.last_frame_time > 0:
            status["last_frame_timestamp"] = datetime.fromtimestamp(
                camera.last_frame_time
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    return jsonify(status)


@app.route('/request_images', methods=['GET', 'POST'])
def request_images():
    """核心接口：修改为返回output目录固定文件的Base64数据（非临时文件）"""
    global camera
    if not camera or not camera.initialized or not camera.running:
        return jsonify({
            "success": False,
            "error": "相机未初始化或已停止",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500
    
    # 检查output目录的固定文件是否存在（核心修改：不再检查临时文件）
    output_files = {
        "color": camera.output_color_path,
        "depth": camera.output_depth_path,
        "depth_vis": camera.output_depth_vis_path
    }
    for img_type, path in output_files.items():
        if not os.path.exists(path):
            return jsonify({
                "success": False,
                "error": f"output目录的{img_type}文件尚未生成（可能服务刚启动，未完成10帧处理）",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "suggestion": "等待1-2秒（15FPS下10帧约需0.7秒）后重新请求"
            }), 404
    
    try:
        # 1. 接口触发保存：从output复制文件到request_saved（修改后逻辑）
        req_save_result = camera.save_request_frames()
        
        # 2. 读取output固定文件并编码为Base64（核心修改：数据源从临时文件改为output文件）
        def encode_img(path, is_depth=False):
            with open(path, 'rb') as f:
                img_data = f.read()
            # 验证深度图是否为16位（可选，增强鲁棒性）
            if is_depth:
                temp_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if temp_img.dtype != np.uint16:
                    print(f"警告：{path} 不是16位深度图，可能存在异常")
            return base64.b64encode(img_data).decode('utf-8')
        
        # 编码output文件（depth标记为16位）
        color_base64 = encode_img(output_files["color"])
        depth_base64 = encode_img(output_files["depth"], is_depth=True)
        depth_vis_base64 = encode_img(output_files["depth_vis"])
        
        # 3. 构造返回结果（明确标注数据源为output固定文件）
        return jsonify({
            "success": True,
            "request_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "request_save_status": {
                "success": req_save_result,
                "save_dir": camera.req_save_root,
                "note": "从output目录复制文件，与自动更新的内容一致"
            },
            "color_image": {
                "format": "PNG (8位BGR)",
                "resolution": f"{camera.color_intrinsics.width}x{camera.color_intrinsics.height}" if (camera and camera.color_intrinsics) else "640x480",
                "data_base64": color_base64,
                "source_path": output_files["color"],
                "note": "来自output目录，每10帧更新一次"
            },
            "depth_image": {
                "format": "PNG (16位无符号整数，单位：毫米)",
                "resolution": f"{camera.depth_intrinsics.width}x{camera.depth_intrinsics.height}" if (camera and camera.depth_intrinsics) else "640x480",
                "data_base64": depth_base64,
                "source_path": output_files["depth"],
                "note": "来自output目录，10帧中位数滤波后结果，每10帧更新一次"
            },
            "depth_vis_image": {
                "format": "PNG (8位彩色)",
                "resolution": f"{camera.color_intrinsics.width}x{camera.color_intrinsics.height}" if (camera and camera.color_intrinsics) else "640x480",
                "data_base64": depth_vis_base64,
                "source_path": output_files["depth_vis"],
                "note": "来自output目录，与彩色图同步更新"
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"图像读取/编码失败：{str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500


# 辅助接口：修改为返回output目录固定文件（非临时文件）
@app.route('/color')
def get_color_image():
    """预览output目录的彩色图（color_image.png）"""
    global camera
    if not camera or not camera.initialized or not camera.running:
        return jsonify({"error": "相机未就绪，无法获取彩色图"}), 500
    
    # 检查output彩色文件是否存在
    if not os.path.exists(camera.output_color_path):
        return jsonify({"error": "output目录彩色文件尚未生成，请等待1-2秒"}), 404
    
    # 读取并返回output文件
    with open(camera.output_color_path, 'rb') as f:
        img_data = f.read()
    return Response(
        img_data,
        mimetype='image/png',
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Image-Source": camera.output_color_path,
            "X-Image-Note": "每10帧更新一次的固定文件"
        }
    )


@app.route('/depth')
def get_depth_image():
    """下载output目录的滤波深度图（depth_filtered_image.png）"""
    global camera
    if not camera or not camera.initialized or not camera.running:
        return jsonify({"error": "相机未就绪，无法获取深度图"}), 500
    
    # 检查output深度文件是否存在
    if not os.path.exists(camera.output_depth_path):
        return jsonify({"error": "output目录滤波深度文件尚未生成，请等待1-2秒"}), 404
    
    # 读取并返回output文件（16位）
    with open(camera.output_depth_path, 'rb') as f:
        img_data = f.read()
    return Response(
        img_data,
        mimetype='image/png',
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Content-Disposition": "attachment; filename=depth_filtered_image.png",
            "X-Image-Source": camera.output_depth_path,
            "X-Image-Note": "16位深度图，每10帧更新一次"
        }
    )


@app.route('/depth_vis')
def get_depth_vis_image():
    """预览output目录的深度可视化图（depth_vis_image.png）"""
    global camera
    if not camera or not camera.initialized or not camera.running:
        return jsonify({"error": "相机未就绪，无法获取深度可视化图"}), 500
    
    # 检查output可视化文件是否存在
    if not os.path.exists(camera.output_depth_vis_path):
        return jsonify({"error": "output目录深度可视化文件尚未生成，请等待1-2秒"}), 404
    
    # 读取并返回output文件
    with open(camera.output_depth_vis_path, 'rb') as f:
        img_data = f.read()
    return Response(
        img_data,
        mimetype='image/png',
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Image-Source": camera.output_depth_vis_path,
            "X-Image-Note": "每10帧更新一次的固定文件"
        }
    )


@app.route('/restart')
def restart_camera():
    """重启相机（原有逻辑，新增清理提示）"""
    global camera
    print("收到相机重启指令...")
    try:
        if camera:
            camera.stop()
            time.sleep(1)
        success = init_camera_safe()
        return jsonify({
            "success": success,
            "message": "相机重启成功" if success else "重启失败，请检查硬件连接",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "note": "重启后需等待10帧处理（约0.7秒），output文件才会更新"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"重启相机失败：{str(e)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500


# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    flask_thread = None
    try:
        print("=== Intel D435 相机服务启动（接口返回output固定文件）===")
        if not init_camera_safe():
            print("相机初始化失败，服务无法启动，程序退出")
            exit(1)
        
        flask_thread = threading.Thread(target=run_flask_server, daemon=False)
        flask_thread.start()
        
        print("\n服务启动完成！")
        print("1. 每10帧自动更新 output/intelD435/ 下的固定文件名文件")
        print("2. /request_images 接口返回上述output文件的Base64数据")
        print("3. 接口调用时同步复制output文件到 request_saved 目录")
        print("提示：服务刚启动时，需等待10帧处理（约0.7秒），接口才能返回有效数据")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n用户中断服务，正在清理资源...")
    finally:
        if camera:
            camera.stop()
        if flask_thread and flask_thread.is_alive():
            print("等待Flask服务线程退出...")
        print("所有资源已清理，程序完全退出")