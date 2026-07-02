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

#### 🎬 [主流程 1](./demo/主流程1.mp4)

>    首先导航至拍摄位，获取RGB-D图，进行Groundsam识别分割，生成点云，算出各物体在世界坐标系（原点固定房间左上角）下的坐标并记录，依次驱动移动arm底盘前往各物体前方。
>    到达后，进行近点目标识别，算出物体在机械臂基底坐标系（原点在arm底部）的坐标，然后以当前机械臂末端法兰盘为平面，作目标中心点对于此平面的垂线，然后直线插值移动至物体（先往一侧平移到达垂足，再直线前往，防止碰撞货架）。
>  抓取完原路返回，呼叫小车并放置，小车导航至固定arm工位，固定arm识别抓取放置到传送带上，再二次定位抓取，放回小车，运回移动arm旁。

---

#### 🎬 [主流程 2](./demo/主流程2.mp4)

> 再由移动arm识别抓取物体，经过标准化位置、最初拍摄位置，最后通过之前计算的插值路径原路返回，放回货架原位。

---

### 二、随机位置抓取功能

<table>
<tr>
<td align="center"><b>🎬 [随机位置抓取演示 1](https://github.com/user-attachments/assets/ad20ca20-c4b9-4816-b55e-d42938ad1096)</b></td>
<td align="center"><b>🎬 [随机位置抓取演示 2](https://github.com/user-attachments/assets/680b0e8b-f187-41c1-ae9a-780286e3b669)</b></td>
</tr>
</table>

> 物体可以在货架上任意位置。根据：底盘在世界坐标系中的位置(x1, y1)，底盘旋转角度theta， 底盘到机械臂基底的距离d，机械臂基底相对于机身的初始偏角 beta，算出目标物体在世界导航坐标系中的位置。

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
