<div align="center">

# 🤖 基于多模态感知的双机械臂协同货物分拣流程

**Multimodal Perception-Based Dual-Arm Collaborative Cargo Sorting Process**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📹 演示视频

### 一、主流程演示

#### 🎬 主流程 1

<!-- 主流程1.mp4 待上传 -->

> 展示双机械臂协同分拣系统的完整运行流程，包括视觉感知、目标识别、双臂协同抓取与放置等核心环节。

---

#### 🎬 主流程 2

<!-- 主流程2.mp4 待上传 -->

> 展示系统的第二阶段运行流程，进一步验证双臂协同分拣的稳定性与鲁棒性。

---

### 二、随机位置抓取功能

<table>
<tr>
<td align="center"><b>随机位置抓取演示 1</b></td>
<td align="center"><b>随机位置抓取演示 2</b></td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/ad20ca20-c4b9-4816-b55e-d42938ad1096

</td>
<td>

https://github.com/user-attachments/assets/680b0e8b-f187-41c1-ae9a-780286e3b669

</td>
</tr>
</table>

> 展示系统对不同随机位置物体的抓取能力，验证视觉定位与机械臂运动规划的灵活性与准确性。

---

### 三、系统流程图

<div align="center">

📄 [点击查看完整流程图 (PDF)](https://github.com/user-attachments/files/29586692/1.pdf)

</div>

---

## 📋 项目简介

本项目实现了一套**基于多模态感知的双机械臂协同货物分拣系统**，融合视觉、点云等多种传感器数据，实现对多种货物的自动识别、定位与分拣。

### 核心功能

- 🎯 **多模态感知**：融合 RGB 图像与深度点云，实现精准目标检测与位姿估计
- 🦾 **双臂协同**：两台机械臂协作完成抓取与分拣任务
- 🚗 **移动底盘**：支持自主导航，扩展工作范围
- 🧠 **智能分拣**：基于 Grounded-SAM 模型，支持多种物体识别

---

## 🛠️ 环境依赖

```bash
pip install opencv-python open3d numpy
```

---

## 🚀 使用方法

### 1. 启动服务

**移动机械臂（arm1）：**
```bash
ssh woosh@192.168.1.226
cd catkin_ws/src/demo/scripts/LT_flow_error
python http_robot_gripper.py    # arm1 控制服务
python http_camera1.py          # camera1 服务
```

**固定机械臂（arm2）：**
```bash
ssh iuucb@192.168.1.231
cd LT/
python http_gripper2.py         # arm2 控制服务
python http_camera2.py          # camera2 服务
```

### 2. 运行主程序

```bash
python main4-多物体循环.py
```

---

## 📁 项目结构

```
├── GroundSam/              # Grounded-SAM 目标检测模块
│   ├── GroundSam_底层/     # 底层 HTTP 接口
│   └── GroundSam_外层/     # 外层调用封装
├── arm1/                   # 移动机械臂控制模块
├── arm2/                   # 固定机械臂控制模块
├── 相机/                   # 相机控制模块
├── 移动底盘/               # 移动底盘控制模块
├── 其他工具/               # ICP配准、点云处理等工具
└── main4-多物体循环.py     # 主程序
```

---

<div align="center">

**⭐ 如果觉得有用，欢迎 Star！⭐**

</div>
