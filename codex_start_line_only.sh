#!/bin/bash
set -e
bash /userdata/codex_stop_all.sh || true

# 1. bringup (chassis + camera + lidar)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch originbot_bringup originbot.launch.py use_camera:=true use_lidar:=true' >/userdata/codex_originbot.log 2>&1 &
sleep 8

# 2. codec (decode image -> shared memory)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch hobot_codec hobot_codec_decode.launch.py codec_sub_topic:=/image' >/userdata/codex_codec.log 2>&1 &
sleep 3

# 3. YOLO traffic light detection
nohup bash -lc 'source /opt/tros/humble/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run dnn_node_example example --ros-args --log-level warn -p feed_type:=1 -p dump_render_img:=0 -p is_shared_mem_sub:=1 -p config_file:=/userdata/yolo11/yolov11workconfig.json -p msg_pub_topic_name:=traffice_sign' >/userdata/codex_yolo.log 2>&1 &
sleep 2

# 4. obstacle detection (LiDAR -> /obstacle_info)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run lanedetection obstacle_detection' >/userdata/codex_obstacle.log 2>&1 &
sleep 1

# 5. line follower perception (ResNet18 on BPU)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run line_follower_perception line_follower_perception --ros-args -p model_path:=/userdata/resnet18/resnet18_224x224_nv12.bin -p model_name:=resnet18_224x224_nv12' >/userdata/codex_perception.log 2>&1 &

# 6. visualization
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run line_follower_perception line_follower_visualization' >/userdata/codex_visualization.log 2>&1 &

# 7. lanedetection (PID tracking + avoidance + traffic + finish)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run lanedetection lanedetection' >/userdata/codex_lanedetection.log 2>&1 &
sleep 2

# 8. websocket (debug view)
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch websocket websocket.launch.py websocket_only_show_image:=true websocket_image_topic:=/image_marked websocket_image_type:=mjpeg' >/userdata/codex_websocket_marked.log 2>&1 &
