#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ai_msgs.msg import PerceptionTargets

class ViewTraffic(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("Traffic_sign is OK")

        self.traffic_sub= self.create_subscription(PerceptionTargets, 
            "/traffice_sign", self.traffic_callback, 10)       
    
    def traffic_callback(self, msg):         
         for boxes in msg.targets:
            if boxes.rois[0].confidence > 0.3 and boxes.rois[0].type == 'green_light\r':                
                print('green_light')
            elif boxes.rois[0].confidence > 0.3 and boxes.rois[0].type == 'red_light':                
                print('red_light')                  

def main(args=None):
    rclpy.init(args=args)        
    trafficview = ViewTraffic("View_Traffic")
    rclpy.spin(trafficview)
    trafficview.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
