import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import time

# 添加项目路径
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO相关导入
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything相关导入
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

# 配置Flask应用
app = Flask(__name__)

# 确保中文正常显示
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型配置
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
USE_SAM_HQ = False
SAM_HQ_CHECKPOINT = None
BERT_BASE_UNCASED_PATH = None

# 全局模型变量
groundingdino_model = None
sam_predictor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 全局根目录（所有请求目录的父目录，服务启动时创建）
root_dir = None
# 全局日志（记录所有请求概况）
global_logger = None


def setup_global_env():
    """服务启动时初始化：创建全局根目录 + 全局日志"""
    global root_dir, global_logger
    
    # 根目录命名：service_启动时间（区分不同服务实例）
    root_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("result", f"service_{root_timestamp}")
    os.makedirs(root_dir, exist_ok=True)
    
    # 初始化全局日志
    global_logger = logging.getLogger("global_logger")
    global_logger.setLevel(logging.INFO)
    
    # 全局日志处理器：控制台 + 全局日志文件
    global_log_file = os.path.join(root_dir, "global_service_log.txt")
    file_handler = logging.FileHandler(global_log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # 日志格式
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)
    
    # 添加处理器（避免重复添加）
    if not global_logger.handlers:
        global_logger.addHandler(file_handler)
        global_logger.addHandler(stream_handler)
    
    global_logger.info(f"服务全局环境初始化完成，根目录：{root_dir}")
    return root_dir


def create_request_dir():
    """为当前请求创建独立目录（含子目录）+ 请求专属日志"""
    # 请求目录命名：request_请求时间（精确到毫秒，避免并发冲突）
    req_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    request_dir = os.path.join(root_dir, f"request_{req_timestamp}")
    
    # 创建请求目录下的子目录（原始图、结果图、掩码）
    subdirs = {
        "ori_image": os.path.join(request_dir, "ori_image"),  # 单请求1张原图，目录名用单数
        "outputs": os.path.join(request_dir, "outputs"),
        "masks": os.path.join(request_dir, "masks")
    }
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 初始化请求专属日志（记录单请求详细过程）
    req_logger = logging.getLogger(f"req_logger_{req_timestamp}")
    req_logger.setLevel(logging.INFO)
    req_log_file = os.path.join(request_dir, "request_detail_log.txt")
    req_file_handler = logging.FileHandler(req_log_file, encoding="utf-8")
    req_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    if not req_logger.handlers:
        req_logger.addHandler(req_file_handler)
    
    global_logger.info(f"请求独立目录创建完成：{request_dir}")
    req_logger.info(f"请求目录初始化完成，子目录：{list(subdirs.keys())}")
    
    return request_dir, subdirs, req_logger


def log_request_info(req_logger, request, text_prompt, box_threshold, text_threshold):
    """记录请求信息（同时写入全局日志和请求专属日志）"""
    log_msg = [
        f"收到请求 - 客户端IP：{request.remote_addr}",
        f"文本提示：{text_prompt}",
        f"框阈值：{box_threshold}，文本阈值：{text_threshold}"
    ]
    for msg in log_msg:
        global_logger.info(msg)
        req_logger.info(msg)


def log_processing_result(req_logger, result_count, processing_time):
    """记录处理结果（同时写入全局日志和请求专属日志）"""
    if result_count > 0:
        log_msg = f"处理完成 - 检测到{result_count}个对象，耗时{processing_time:.2f}秒"
    else:
        log_msg = f"处理完成 - 未检测到对象，耗时{processing_time:.2f}秒"
    global_logger.info(log_msg)
    req_logger.info(log_msg)


def load_image(image_data):
    """从字节数据加载图像（无修改）"""
    image_pil = Image.open(BytesIO(image_data)).convert("RGB")
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_models():
    """初始化GroundingDINO和SAM模型（无修改，仅调整日志输出）"""
    global groundingdino_model, sam_predictor
    
    global_logger.info("开始加载模型...")
    
    # 加载GroundingDINO模型
    args = SLConfig.fromfile(GROUNDING_DINO_CONFIG_PATH)
    args.device = device
    args.bert_base_uncased_path = BERT_BASE_UNCASED_PATH
    groundingdino_model = build_model(args)
    checkpoint = torch.load(GROUNDING_DINO_WEIGHTS_PATH, weights_only=True, map_location="cpu")
    load_res = groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    groundingdino_model = groundingdino_model.to(device)
    groundingdino_model.eval()
    global_logger.info(f"GroundingDINO加载完成，结果：{load_res}")
    
    # 加载SAM模型
    if USE_SAM_HQ:
        sam = sam_hq_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_HQ_CHECKPOINT)
    else:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    
    sam = sam.to(device)
    sam_predictor = SamPredictor(sam)
    global_logger.info("SAM模型加载完成")
    
    global_logger.info("所有模型初始化完成")


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """GroundingDINO推理（无修改）"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # 过滤输出
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # 获取预测短语
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []

    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(f"{pred_phrase}({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    """绘制掩码（无修改）"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """绘制边界框（无修改）"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(mask_dir, mask_list, box_list, label_list, req_logger):
    """保存掩码数据（路径改为请求专属的masks目录）"""
    value = 0  # 0表示背景
    mask_img = torch.zeros(mask_list.shape[-2:])
    
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    # 保存掩码合并图
    mask_merge_path = os.path.join(mask_dir, "mask_merge.jpg")
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(mask_merge_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # 保存掩码JSON（含标签、置信度、边界框）
    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        if '(' in label and ')' in label:
            name, logit = label.split('(')
            logit = float(logit[:-1])  # 去除末尾的')'
        else:
            name = label
            logit = 0.0
        json_data.append({
            "value": value,
            "label": name,
            "logit": logit,
            "box": box.numpy().tolist()
        })
    
    mask_json_path = os.path.join(mask_dir, "mask_info.json")
    with open(mask_json_path, 'w', encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    req_logger.info(f"掩码数据保存完成：合并图{mask_merge_path}，信息JSON{mask_json_path}")
    return mask_merge_path, mask_json_path


def annotate_image(image, boxes, labels, color=(0, 255, 0), thickness=2):
    """使用OpenCV标注图像（无修改）"""
    annotated = image.copy()
    h, w, _ = annotated.shape

    for box, label in zip(boxes, labels):
        # 将相对坐标转换为绝对坐标
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)

        # 绘制边界框
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, thickness)

        # 准备标签文本
        if '(' in label and ')' in label:
            name, logit = label.split('(')
            logit = logit[:-1]  # 去除末尾的')'
            label_text = f"{name}: {logit}"
        else:
            label_text = label

        # 绘制标签背景
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (xmin, ymin - label_height - 10), (xmin + label_width, ymin), color, -1)

        # 绘制标签文本
        cv2.putText(annotated, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated


@app.route('/process', methods=['POST'])
def process_request():
    """处理请求（核心修改：基于请求专属目录保存文件）"""
    start_time = time.time()
    req_logger = None  # 避免异常时变量未定义
    try:
        # 1. 检查请求参数
        if 'image' not in request.files or 'text_prompt' not in request.form:
            error_msg = "请求参数缺失：需包含'image'文件和'text_prompt'文本"
            global_logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # 2. 创建当前请求的独立目录 + 专属日志
        request_dir, subdirs, req_logger = create_request_dir()
        ori_img_dir = subdirs["ori_image"]
        outputs_dir = subdirs["outputs"]
        masks_dir = subdirs["masks"]
        
        # 3. 获取请求参数
        image_file = request.files['image']
        image_data = image_file.read()
        text_prompt = request.form['text_prompt']
        box_threshold = float(request.form.get('box_threshold', 0.3))
        text_threshold = float(request.form.get('text_threshold', 0.25))
        
        # 4. 记录请求基本信息
        log_request_info(req_logger, request, text_prompt, box_threshold, text_threshold)
        
        # 5. 保存原始图像（请求专属目录下）
        raw_image_path = os.path.join(ori_img_dir, "raw_image.jpg")  # 单图无需时间戳后缀
        with open(raw_image_path, "wb") as f:
            f.write(image_data)
        req_logger.info(f"原始图像保存路径：{raw_image_path}")
        
        # 6. 加载图像并推理
        image_pil, image_tensor = load_image(image_data)
        boxes_filt, pred_phrases = get_grounding_output(
            groundingdino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=device
        )
        
        # 7. 无检测结果的处理
        if boxes_filt.size(0) == 0:
            processing_time = time.time() - start_time
            log_processing_result(req_logger, 0, processing_time)
            return jsonify({
                'message': '未检测到任何目标',
                'request_dir': request_dir,  # 返回请求目录，方便用户查找
                'objects': []
            }), 200
        
        # 8. SAM分割推理
        image_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_cv_rgb)
        
        # 处理边界框坐标（转换为绝对坐标）
        W, H = image_pil.size
        boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])  # 相对坐标转绝对坐标
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2  # (x_center,y_center,w,h) -> (xmin,ymin,w,h)
        boxes_filt[:, 2:] += boxes_filt[:, :2]  # -> (xmin,ymin,xmax,ymax)
        boxes_filt = boxes_filt.cpu()
        
        # SAM预测掩码
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv_rgb.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # 9. 保存结果图像（请求专属outputs目录）
        # 9.1 Grounded-SAM可视化结果
        result_vis_path = os.path.join(outputs_dir, "grounded_sam_result.jpg")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv_rgb)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(result_vis_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()
        req_logger.info(f"Grounded-SAM结果图保存：{result_vis_path}")
        
        # 9.2 OpenCV标注结果（便于快速查看）
        annotated_image = annotate_image(
            image_cv,  # 原始BGR图像（OpenCV默认格式）
            boxes_filt.numpy() / np.array([W, H, W, H]),  # 转回相对坐标用于标注
            pred_phrases
        )
        annotated_path = os.path.join(outputs_dir, "annotated_result.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        req_logger.info(f"OpenCV标注图保存：{annotated_path}")
        
        # 10. 保存掩码数据（请求专属masks目录）
        mask_merge_path, mask_json_path = save_mask_data(masks_dir, masks, boxes_filt, pred_phrases, req_logger)
        
        # 11. 保存单个对象的掩码（便于单独使用）
        single_mask_paths = []
        for i, (mask, label) in enumerate(zip(masks, pred_phrases)):
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            # 提取标签名作为掩码文件名（去除置信度）
            if '(' in label and ')' in label:
                label_name = label.split('(')[0].strip()
            else:
                label_name = label
            single_mask_path = os.path.join(masks_dir, f"mask_{i+1}_{label_name}.png")
            cv2.imwrite(single_mask_path, mask_np)
            single_mask_paths.append(single_mask_path)
            req_logger.info(f"单个掩码保存：{single_mask_path}")
        
        # 12. 转换结果图像为Base64（便于前端直接显示）
        def img_to_base64(img_path):
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        result_img_b64 = img_to_base64(result_vis_path)
        annotated_img_b64 = img_to_base64(annotated_path)
        
        # 13. 整理响应数据
        objects = []
        for i, (box, label, mask, single_path) in enumerate(zip(boxes_filt, pred_phrases, masks, single_mask_paths)):
            # 解析标签和置信度
            if '(' in label and ')' in label:
                name, logit = label.split('(')
                logit = float(logit[:-1])
            else:
                name = label
                logit = 0.0
            
            # 单个掩码Base64
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            _, mask_buf = cv2.imencode('.png', mask_np)
            mask_b64 = base64.b64encode(mask_buf).decode('utf-8')
            
            objects.append({
                'id': i + 1,
                'label': name,
                'confidence': logit,
                'box': {
                    'xmin': float(box[0] / W),  # 返回相对坐标（便于跨分辨率使用）
                    'ymin': float(box[1] / H),
                    'xmax': float(box[2] / W),
                    'ymax': float(box[3] / H)
                },
                'mask': mask_b64,
                'mask_path': single_path
            })
        
        # 14. 记录处理结果
        processing_time = time.time() - start_time
        log_processing_result(req_logger, len(objects), processing_time)
        
        # 15. 返回响应（包含请求目录路径，便于用户本地查找文件）
        return jsonify({
            'message': '处理成功',
            'request_dir': request_dir,  # 关键：返回当前请求的目录路径
            'detected_objects': len(objects),
            'objects': objects,
            'result_image': {
                'grounded_sam': result_img_b64,
                'annotated': annotated_img_b64
            },
            'file_paths': {
                'raw_image': raw_image_path,
                'grounded_sam_result': result_vis_path,
                'annotated_result': annotated_path,
                'mask_merge': mask_merge_path,
                'mask_info_json': mask_json_path
            }
        }), 200
    
    except Exception as e:
        # 异常处理：同时记录全局和请求日志
        error_msg = f"请求处理失败：{str(e)}"
        if req_logger:
            req_logger.error(error_msg, exc_info=True)
        global_logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查（返回全局根目录和模型状态）"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'models_loaded': (groundingdino_model is not None) and (sam_predictor is not None),
        'global_root_dir': root_dir,
        'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200


if __name__ == '__main__':
    # 服务启动流程：初始化全局环境 -> 加载模型 -> 启动服务
    setup_global_env()
    load_models()
    global_logger.info("服务启动：http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)  # 开启线程支持多请求
