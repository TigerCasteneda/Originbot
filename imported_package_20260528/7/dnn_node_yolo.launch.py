import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    dnn_node = Node(
        package='dnn_node_example',
        executable='example',
        name='dnn_node',
        output='screen',
        arguments=[
            '--ros-args', '--log-level', 'warn',
            '--ros-args', '-p', 'feed_type:=1',
            '--ros-args', '-p', 'dump_render_img:=0',
            '--ros-args', '-p', 'is_shared_mem_sub:=1',
            '--ros-args', '-p', 'config_file:=/userdata/yolo11/yolov11workconfig.json',
            '--ros-args', '-p', 'msg_pub_topic_name:=traffice_sign'
        ]
    )
    return LaunchDescription([
        dnn_node,
    ])
