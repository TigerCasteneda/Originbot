#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy, cv2, cv_bridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import uuid 
import os

class Picker(Node):
    def __init__(self):
        super().__init__('end2end_image_recorder')
        self.get_logger().info("start picking!")
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = self.create_subscription(CompressedImage, '/image', self.image_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.num=0

    
    def image_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        

    def cmd_vel_callback(self, msg):
        # 记录每一次收到的指令＆当前采集到的画面
        self.num +=1
        if self.num ==5:     
            print("Hello")
            img_name = 'zx_%f_%f_%s' % (msg.angular.z,msg.linear.x, uuid.uuid1())   
            self.image = self.image[120:456, :, :]    #取车道线兴趣区域
            cv2.imwrite('/userdata/pictures/image_dataset/' + img_name + '.jpg', self.image)
            print('image saved!')
            self.num =0        

def main(args=None):
    rclpy.init(args=args)    
    picker = Picker()
    try:
        rclpy.spin(picker)
    except KeyboardInterrupt:
        pass
    finally:

        picker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
