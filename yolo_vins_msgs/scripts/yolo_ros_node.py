#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from yolo_vins_msgs.msg import BoundingBox, BoundingBoxes
from ultralytics import YOLO

class YoloRosNode:
    def __init__(self):
        rospy.init_node('yolo_ros_node', anonymous=True)
        
        rospy.loginfo("Loading YOLO model...")
        # 加载官方 YOLO 模型 
        self.model = YOLO("yolo26n.pt") # 这里填你的模型

        self.bbox_pub = rospy.Publisher("/yolo/bounding_boxes", BoundingBoxes, queue_size=10)
        
        # KITTI 数据集通常发布在 /kitti/camera_color_left/image_raw
        image_topic = rospy.get_param("~image_topic", "/kitti/camera_color_left/image_raw")
        
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("YOLO Node Initialized. Listening to %s", image_topic)

    # 【核心黑客技巧】纯 numpy 解析 ROS Image，彻底甩掉 cv_bridge 的 Python2 依赖
    def imgmsg_to_cv2(self, img_msg):
        # KITTI 和大多数相机通常是 rgb8 或 bgr8 编码
        dtype = np.uint8
        channels = 3
        # 直接从底层内存 buffer 读取字节并重构矩阵维度
        image_np = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, channels)
        
        # ROS 里的 rgb8 需要转成 OpenCV 默认的 bgr8
        if img_msg.encoding == "rgb8":
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np

    def image_callback(self, img_msg):
        # 1. 极速内存反序列化
        cv_image = self.imgmsg_to_cv2(img_msg)
            
        # 2. 执行 YOLO 推理
        results = self.model(cv_image, verbose=False)
        
        # 3. 初始化跨进程通信消息
        bbox_msg = BoundingBoxes()
        
        # 时间戳绝对对齐 (成败的关键)
        bbox_msg.header.stamp = img_msg.header.stamp 
        bbox_msg.header.frame_id = img_msg.header.frame_id
        
        # 4. 解析推理结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                #从 YOLO 结果中提取出置信度
                conf = box.conf[0].cpu().item()
                cls_id = int(box.cls[0].cpu().item())
                
                # 过滤动态物体 (COCO: 0=行人, 2=汽车, 3=摩托, 5=公交, 7=卡车)
                if cls_id in [0, 2, 3, 5, 7]:
                    b = BoundingBox()
                    b.xmin = int(xyxy[0])
                    b.ymin = int(xyxy[1])
                    b.xmax = int(xyxy[2])
                    b.ymax = int(xyxy[3])
                    #将置信度塞进 ROS 消息体中
                    b.probability = float(conf)  
                    b.id = int(cls_id)
                    bbox_msg.bounding_boxes.append(b)
                    
        # 5. 打包发送给 VINS 前端
        self.bbox_pub.publish(bbox_msg)

if __name__ == '__main__':
    try:
        node = YoloRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass