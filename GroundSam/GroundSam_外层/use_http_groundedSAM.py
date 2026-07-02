import requests
import base64
import cv2
import numpy as np
import json
import os
# 新增：导入时间模块用于生成时间戳
import datetime


def call_grounded_sam_service(
    service_url: str,
    image: np.ndarray,  # 注：原函数参数名"image_path"与内部逻辑（接收numpy图像）冲突，已修正为"image"（见下方说明）
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    save_result_images: bool = True,
    save_masks: bool = True,
    display_results: bool = False,
    save_dir: str = "output/res_groundedSAM"
) -> dict:
    """
    调用Grounded-SAM服务的核心函数，返回识别框、掩码数据（顺序一致）及处理结果

    Args:
        service_url (str): Grounded-SAM服务地址（如 "http://192.168.1.9:1236/process"）
        image (np.ndarray): 待检测的图像numpy数组（OpenCV读取的BGR格式）
        text_prompt (str): 文本提示词（多个目标用"."分隔，如 "a chair. a cup."）
        box_threshold (float, optional): 边界框置信度阈值. Defaults to 0.35.
        text_threshold (float, optional): 文本匹配阈值. Defaults to 0.25.
        save_result_images (bool, optional): 是否保存服务返回的标注结果图像. Defaults to True.
        save_masks (bool, optional): 是否保存每个检测对象的掩码图像文件. Defaults to True.
        display_results (bool, optional): 是否弹窗显示结果图像. Defaults to False.
        save_dir (str, optional): 结果根目录（每次调用会在该目录下生成res_时间子目录）. Defaults to "output/res_groundedSAM".

    Returns:
        dict: 服务调用结果字典，包含以下键：
            - "status": 调用状态（"success"/"fail"）
            - "message": 状态描述（成功信息/错误原因）
            - "response_data": 服务返回的原始JSON数据（status=success时有效）
            - "detected_objects": 检测到的对象总数（status=success时有效）
            - "detected_boxes": 识别框列表（status=success时有效，与detected_masks顺序一致），每个元素为：
                {
                    "id": 目标ID,
                    "label": 目标标签,
                    "confidence": 置信度,
                    "box": {
                        "xmin": 相对坐标x最小值,
                        "ymin": 相对坐标y最小值,
                        "xmax": 相对坐标x最大值,
                        "ymax": 相对坐标y最大值
                    }
                }
            - "detected_masks": 掩码数据列表（status=success时有效，与detected_boxes顺序一致），每个元素为：
                {
                    "id": 目标ID（与对应box的ID一致）,
                    "label": 目标标签（与对应box的标签一致）,
                    "confidence": 置信度（与对应box的置信度一致）,
                    "mask_data": 掩码numpy数组（单通道，shape=(H,W)，值为0/255，0=背景，255=目标）,
                    "mask_save_path": 掩码文件保存路径（save_masks=True时有效，否则为None）
                }
            - "saved_images": 结果图像保存路径列表（status=success且save_result_images=True时有效）
    """
    # 初始化返回结果（确保结构统一，即使失败也有完整键）
    result = {
        "status": "fail",
        "message": "",
        "response_data": None,
        "detected_objects": 0,
        "detected_boxes": [],  # 新增：识别框列表
        "detected_masks": [],  # 新增：掩码数据列表（与boxes顺序一致）
        "saved_images": []
    }

    # -------------------------- 1. 输入合法性检查 --------------------------
    # 修正：原代码"image_path"参数与内部"读取numpy图像"逻辑冲突，已改为检查numpy数组有效性
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        result["message"] = f"输入图像需为OpenCV读取的BGR格式numpy数组（当前类型：{type(image)}，维度：{image.ndim if isinstance(image, np.ndarray) else '无'}）"
        return result

    # 检查服务地址格式
    if not (service_url.startswith("http://") or service_url.startswith("https://")):
        result["message"] = f"服务地址格式无效：{service_url}（需以http://或https://开头）"
        return result

    # 检查阈值范围（0~1）
    if not (0 <= box_threshold <= 1):
        result["message"] = f"边界框阈值需在0~1之间，当前值：{box_threshold}"
        return result
    if not (0 <= text_threshold <= 1):
        result["message"] = f"文本阈值需在0~1之间，当前值：{text_threshold}"
        return result

    # -------------------------- 2. 创建保存目录（核心修改：自动生成res_时间子目录） --------------------------
    # 生成时间戳（格式：YYYYMMDD_HHMMSS，精确到秒，避免同名冲突）
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 构建带时间戳的子目录（格式：save_dir/res_YYYYMMDD_HHMMSS）
    timestamp_save_dir = os.path.join(save_dir, f"res_{time_stamp}")
    # 掩码子目录（在时间戳目录下）
    mask_dir = os.path.join(timestamp_save_dir, "masks") if (save_masks or save_result_images) else None

    if (save_masks or save_result_images):
        try:
            # 创建时间戳主目录和掩码子目录（exist_ok=True避免目录已存在报错）
            os.makedirs(timestamp_save_dir, exist_ok=True)
            if save_masks:
                os.makedirs(mask_dir, exist_ok=True)
            print(f"[Grounded-SAM] 结果保存目录已创建：{timestamp_save_dir}")
        except Exception as e:
            result["message"] = f"创建保存目录失败：{str(e)}"
            return result

    # -------------------------- 3. 加载图像并构造请求数据 --------------------------
    # 编码为JPEG二进制（减少传输体积）
    encode_success, image_buffer = cv2.imencode('.jpg', image)
    if not encode_success:
        result["message"] = f"图像编码为JPEG失败"
        return result
    image_file = image_buffer.tobytes()

    # 构造请求参数（files传图像，data传文本参数）
    request_files = {
        "image": ("input_image.jpg", image_file, "image/jpeg")
    }
    request_data = {
        "text_prompt": text_prompt,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold)
    }

    # -------------------------- 4. 发送POST请求到服务 --------------------------
    print(f"[Grounded-SAM] 向服务发送请求：{service_url}")
    print(f"[Grounded-SAM] 文本提示：{text_prompt} | 边界框阈值：{box_threshold} | 文本阈值：{text_threshold}")

    try:
        response = requests.post(
            url=service_url,
            files=request_files,
            data=request_data,
            timeout=30
        )

        # -------------------------- 5. 处理服务响应 --------------------------
        if response.status_code == 200:
            # 解析服务返回的JSON数据
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                result["message"] = f"服务返回数据非JSON格式：{str(e)}"
                return result

            # 更新基础结果信息
            result["status"] = "success"
            result["message"] = "服务调用成功"
            result["response_data"] = response_data
            detected_obj_count = response_data.get("detected_objects", 0)
            result["detected_objects"] = detected_obj_count

            # -------------------------- 5.1 提取识别框与掩码（核心修改：保证顺序一致） --------------------------
            objects = response_data.get("objects", [])
            if detected_obj_count > 0 and len(objects) > 0:
                print(f"[Grounded-SAM] 检测到 {detected_obj_count} 个目标，提取识别框与掩码（顺序保持一致）...")

                for obj in objects:
                    # 1. 提取当前目标的基础信息（ID、标签、置信度）
                    obj_id = obj.get("id", f"obj_{len(result['detected_boxes'])}")  # 兜底ID（避免None）
                    obj_label = obj.get("label", "unknown").strip()
                    obj_conf = round(obj.get("confidence", 0.0), 4)

                    # 2. 提取识别框（相对坐标，服务端通常返回0~1的相对值）
                    obj_box = obj.get("box", {})
                    box_info = {
                        "id": obj_id,
                        "label": obj_label,
                        "confidence": obj_conf,
                        "box": {
                            "xmin": round(obj_box.get("xmin", 0.0), 4),
                            "ymin": round(obj_box.get("ymin", 0.0), 4),
                            "xmax": round(obj_box.get("xmax", 0.0), 4),
                            "ymax": round(obj_box.get("ymax", 0.0), 4)
                        }
                    }
                    # 添加到识别框列表（顺序1）
                    result["detected_boxes"].append(box_info)

                    # 3. 提取掩码数据（Base64解码为numpy数组）
                    mask_data = None
                    mask_save_path = None
                    if "mask" in obj:
                        try:
                            # 解码Base64掩码（服务端返回的掩码通常为单通道0-255格式）
                            mask_b64 = obj["mask"]
                            mask_np = np.frombuffer(base64.b64decode(mask_b64), np.uint8)
                            mask_data = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)  # 单通道数组（H,W）

                            # 验证掩码有效性（避免空数据）
                            if mask_data is None:
                                print(f"[Grounded-SAM] 警告：目标ID={obj_id} 的掩码解码为空")
                        except Exception as e:
                            print(f"[Grounded-SAM] 解码目标ID={obj_id} 的掩码失败：{str(e)}")

                    # 4. 保存掩码文件（路径改为时间戳子目录下的masks文件夹）
                    if save_masks and mask_data is not None:
                        try:
                            # 生成带标识的文件名（确保与识别框对应）
                            label_clean = obj_label.replace(" ", "_").replace(".", "")  # 清理标签特殊字符
                            mask_filename = f"mask_id_{obj_id}_label_{label_clean}_conf_{obj_conf:.2f}.png"
                            mask_save_path = os.path.join(mask_dir, mask_filename)  # 路径指向时间戳子目录的masks

                            # 保存掩码图像（0=背景，255=目标）
                            cv2.imwrite(mask_save_path, mask_data)
                            print(f"[Grounded-SAM] 掩码文件保存：{mask_save_path}")
                        except Exception as e:
                            print(f"[Grounded-SAM] 保存目标ID={obj_id} 的掩码文件失败：{str(e)}")
                            mask_save_path = None  # 保存失败则置空路径

                    # 5. 构造掩码信息（与识别框顺序一致）
                    mask_info = {
                        "id": obj_id,
                        "label": obj_label,
                        "confidence": obj_conf,
                        "mask_data": mask_data,  # numpy数组（可直接用于后续处理，如点云生成）
                        "mask_save_path": mask_save_path  # 文件路径（指向时间戳子目录）
                    }
                    # 添加到掩码列表（顺序2：与识别框列表完全对应）
                    result["detected_masks"].append(mask_info)

                # 验证顺序一致性（防异常）
                if len(result["detected_boxes"]) != len(result["detected_masks"]):
                    print(f"[Grounded-SAM] 警告：识别框数量（{len(result['detected_boxes'])}）与掩码数量（{len(result['detected_masks'])}）不一致！")
            else:
                print(f"[Grounded-SAM] 未检测到任何目标（服务消息：{response_data.get('message', '无')}）")

            # -------------------------- 5.2 处理结果图像（保存路径改为时间戳子目录） --------------------------
            result_image = response_data.get("result_image", {})
            saved_image_paths = []
            if result_image and "grounded_sam" in result_image and "annotated" in result_image:
                # 处理Grounded-SAM可视化图（保存到时间戳子目录）
                try:
                    gs_b64 = result_image["grounded_sam"]
                    gs_np = np.frombuffer(base64.b64decode(gs_b64), np.uint8)
                    gs_img = cv2.imdecode(gs_np, cv2.IMREAD_COLOR)
                    if gs_img is not None and save_result_images:
                        gs_save_path = os.path.join(timestamp_save_dir, "grounded_sam_result.jpg")  # 路径指向时间戳子目录
                        cv2.imwrite(gs_save_path, gs_img)
                        saved_image_paths.append(gs_save_path)
                except Exception as e:
                    print(f"[Grounded-SAM] 处理Grounded-SAM结果图失败：{str(e)}")

                # 处理OpenCV标注图（保存到时间戳子目录）
                try:
                    anno_b64 = result_image["annotated"]
                    anno_np = np.frombuffer(base64.b64decode(anno_b64), np.uint8)
                    anno_img = cv2.imdecode(anno_np, cv2.IMREAD_COLOR)
                    if anno_img is not None and save_result_images:
                        anno_save_path = os.path.join(timestamp_save_dir, "annotated_result.jpg")  # 路径指向时间戳子目录
                        cv2.imwrite(anno_save_path, anno_img)
                        saved_image_paths.append(anno_save_path)
                except Exception as e:
                    print(f"[Grounded-SAM] 处理标注结果图失败：{str(e)}")

                # 可视化结果（如果开启）
                if display_results and (gs_img is not None or anno_img is not None):
                    print(f"[Grounded-SAM] 显示结果图像（按任意键关闭）...")
                    if gs_img is not None:
                        cv2.namedWindow("Grounded-SAM Result", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Grounded-SAM Result", 800, 600)
                        cv2.imshow("Grounded-SAM Result", gs_img)
                    if anno_img is not None:
                        cv2.namedWindow("Annotated Result", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Annotated Result", 800, 600)
                        cv2.imshow("Annotated Result", anno_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            result["saved_images"] = saved_image_paths

        else:
            # 服务返回非200状态码（请求失败）
            result["message"] = f"服务请求失败（HTTP {response.status_code}）"
            try:
                error_detail = response.json().get("error", "未知错误")
                result["message"] += f" | 详情：{error_detail}"
            except:
                result["message"] += f" | 详情：{response.text[:200]}..."  # 截取前200字符避免过长

    except requests.exceptions.RequestException as e:
        # 捕获请求相关异常（连接超时、服务不可达等）
        result["message"] = f"服务请求异常：{str(e)}"
    except Exception as e:
        # 捕获其他未知异常
        result["message"] = f"处理响应时未知错误：{str(e)}"

    # 打印最终状态
    print(f"\n[Grounded-SAM] 调用完成 | 状态：{result['status']} | 检测目标数：{result['detected_objects']}")
    return result


# -------------------------- 客户端调用示例 --------------------------
def client_demo():
    """客户端示例：调用服务并使用返回的识别框和掩码（顺序一致）"""
    # 1. 配置调用参数
    config = {
        "service_url": "http://192.168.1.19:1236/process",
        "image_path": "output/color_2025-11-14_202702_217.png",  # 本地图像路径（用于读取为numpy数组）
        "text_prompt": "a pink block.",  # 检测“椅子”
        "box_threshold": 0.5,
        "text_threshold": 0.25,
        "save_result_images": True,
        "save_masks": True,
        "display_results": False,
        "save_dir": "output/res_groundedSAM"  # 根目录（每次调用会生成res_时间子目录）
    }

    # 2. 读取本地图像为numpy数组（适配函数输入要求）
    try:
        image = cv2.imread(config["image_path"])
        if image is None:
            print(f"[客户端] 无法读取图像文件：{config['image_path']}")
            return
    except Exception as e:
        print(f"[客户端] 读取图像失败：{str(e)}")
        return

    # 3. 调用Grounded-SAM服务（移除image_path参数，传入image数组）
    service_result = call_grounded_sam_service(
        service_url=config["service_url"],
        image=image,
        text_prompt=config["text_prompt"],
        box_threshold=config["box_threshold"],
        text_threshold=config["text_threshold"],
        save_result_images=config["save_result_images"],
        save_masks=config["save_masks"],
        display_results=config["display_results"],
        save_dir=config["save_dir"]
    )

    # 4. 解析结果（重点：使用顺序一致的boxes和masks）
    if service_result["status"] == "success":
        print(f"\n=== 客户端结果解析 ===")
        print(f"1. 基础信息")
        print(f"   - 检测目标总数：{service_result['detected_objects']}")
        print(f"   - 保存的结果图像：{service_result['saved_images']}")

        # 5. 关联使用识别框与掩码（核心：索引一致）
        boxes = service_result["detected_boxes"]
        masks = service_result["detected_masks"]
        if len(boxes) > 0 and len(boxes) == len(masks):
            print(f"\n2. 识别框与掩码关联（顺序一致）")
            for idx, (box, mask) in enumerate(zip(boxes, masks)):
                print(f"\n   目标 {idx+1}（ID：{box['id']}）")
                print(f"   - 标签：{box['label']} | 置信度：{box['confidence']}")
                print(f"   - 识别框（相对坐标）：xmin={box['box']['xmin']}, ymin={box['box']['ymin']}, xmax={box['box']['xmax']}, ymax={box['box']['ymax']}")
                print(f"   - 掩码信息：数据shape={mask['mask_data'].shape if mask['mask_data'] is not None else 'None'} | 保存路径：{mask['mask_save_path']}")

                # 示例1：使用掩码数据生成点云（假设已创建PointCloudGenerator实例）
                # if mask['mask_data'] is not None:
                #     point_cloud = pcd_generator.generate_point_cloud(color_img, depth_img, mask=mask['mask_data'])

                # 示例2：显示单个目标的掩码（调试用）
                # if mask['mask_data'] is not None:
                #     cv2.imshow(f"Mask of {box['label']} (ID:{box['id']})", mask['mask_data'])
                #     cv2.waitKey(0)
                #     cv2.destroyWindow(f"Mask of {box['label']} (ID:{box['id']})")
        else:
            print(f"\n2. 无有效识别框或掩码（数量不匹配）")
    else:
        print(f"\n=== 调用失败 ===")
        print(f"原因：{service_result['message']}")


if __name__ == "__main__":
    client_demo()