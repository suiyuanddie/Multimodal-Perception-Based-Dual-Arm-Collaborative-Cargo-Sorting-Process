环境：
opencv-python
open3d
numpy
等环境


运行前，需要开启特定的服务：

开启移动机械臂后：
ssh woosh@192.168.1.226
cd catkin_ws/src/demo/scripts/LT_flow_error
	arm1相关：http_robot_gripper.py
	camera1相关：http_camera1.py


开启固定机械臂后：
ssh iuucb@192.168.1.231
cd LT/
	arm2相关：http_gripper2.py
	camera2相关：http_camera2.py


之后进入flow.py文件中，测试每个部分的功能。
