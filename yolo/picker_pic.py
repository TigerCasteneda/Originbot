#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import datetime
import os
import sys
import select
import tty
import termios


class Picker(Node):
    def __init__(self):
        super().__init__('picture_recorder')
        self.get_logger().info("Start picking picture! Press Enter to save.")
        
        self.n = 0
        self.save_dir = '/userdata/pictures/JPEGImages'
        self._ensure_dir_exists()
        
        self.bridge = CvBridge()
        self.original_settings = termios.tcgetattr(sys.stdin)
        
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/image', 
            self.image_callback, 
            10
        )

    def _ensure_dir_exists(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            self.get_logger().info(f"Created directory: {self.save_dir}")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
        return key

    def image_callback(self, msg):
        key = self.get_key()
        
        if key == '\r':
            try:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S_%f')
                img_name = f"{timestamp}.jpg"
                img_save_path = os.path.join(self.save_dir, img_name)
                
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
                success = cv2.imwrite(img_save_path, cv_image)
                
                if success:
                    self.n += 1
                    self.get_logger().info(f"Saved picture #{self.n}: {img_name}")
                else:
                    self.get_logger().error(f"Failed to save picture: {img_name}")
                    
            except Exception as e:
                self.get_logger().error(f"Error saving picture: {e}")

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
        self.get_logger().info(f"Total pictures saved: {self.n}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    picker = Picker()
    
    try:
        rclpy.spin(picker)
    except KeyboardInterrupt:
        picker.get_logger().info("Ctrl+C pressed, shutting down...")
    finally:
        picker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()