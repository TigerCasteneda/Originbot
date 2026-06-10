#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
from std_msgs.msg import Bool

        
class LaneDetection(Node):
    def __init__(self):
        super().__init__('lanedetection')
        self.get_logger().info("Start lane keeping.")

        self.pct_sub = self.create_subscription(Twist, '/pct_vel', 
            self.pct_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        #self.obs_sub = self.create_subscription(Bool, '/has_obs',self.obs_callback,10)
        

        self.twist = Twist()   
        self.pct = Twist() 
        #self.has_obs = 0

    def on_shutdown(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0 
        self.cmd_vel_pub.publish(self.twist) 

    #def obs_callback(self, msg):
    #    self.has_obs = msg.data
        
    def pct_callback(self, msg):
        self.pct=msg
        #self.pct.linear.x  推理出来的小车车速
        #self.pct.angular.z 推理出来的小车转角

        #self.twist.linear.x =      控制小车速度
        #self.twist.angular.z =     控制小车转角             
                   
        
        self.cmd_vel_pub.publish(self.twist)   
       
        
     
def main(args=None):
    rclpy.init(args=args)      
    lanedetection = LaneDetection()
    try:
        rclpy.spin(lanedetection)
    except KeyboardInterrupt:
        lanedetection.on_shutdown()
        lanedetection.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
