# Copyright (c) 2024, www.guyuehome.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def include_launch(package, launch_file, launch_arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items())


def generate_launch_description():
    camera_node = include_launch('originbot_bringup', 'camera.launch.py')

    nv12_codec_node = include_launch(
        'hobot_codec',
        'hobot_codec_decode.launch.py',
        {
            'codec_in_mode': 'ros',
            'codec_out_mode': 'shared_mem',
            'codec_sub_topic': '/image',
            'codec_pub_topic': '/hbmem_img'
        })

    perception_node = Node(
        package='line_follower_perception',
        executable='line_follower_perception',
        name='line_follower_perception',
        output='screen',
        parameters=[{
            'model_path': '/userdata/resnet18/resnet18_224x224_nv12.bin',
            'model_name': 'resnet18_224x224_nv12'
        }])

    visualization_node = Node(
        package='line_follower_perception',
        executable='line_follower_visualization',
        name='line_follower_visualization',
        output='screen')

    web_node = include_launch(
        'websocket',
        'websocket.launch.py',
        {
            'websocket_image_topic': '/image_marked',
            'websocket_image_type': 'mjpeg',
            'websocket_only_show_image': 'True'
        })

    return LaunchDescription([
        camera_node,
        nv12_codec_node,
        perception_node,
        visualization_node,
        web_node
    ])
