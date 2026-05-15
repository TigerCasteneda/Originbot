set -e
pkill -f "originbot.launch.py" || true
pkill -f "vp100.launch.py" || true
pkill -f "hobot_codec_decode.launch.py" || true
pkill -f "line_follower_perception" || true
pkill -f "line_follower_visualization" || true
pkill -f "lanedetection" || true
pkill -f "obstacle_detection" || true
pkill -f "websocket.launch.py" || true
pkill -f "/opt/tros/humble/lib/websocket/websocket" || true
sleep 2
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch originbot_bringup originbot.launch.py use_camera:=true use_lidar:=true' >/userdata/codex_originbot_lidar.log 2>&1 &
sleep 8
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch hobot_codec hobot_codec_decode.launch.py codec_sub_topic:=/image' >/userdata/codex_codec.log 2>&1 &
sleep 4
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run line_follower_perception line_follower_perception --ros-args -p model_path:=/userdata/resnet18/resnet18_224x224_nv12.bin -p model_name:=resnet18_224x224_nv12' >/userdata/codex_perception.log 2>&1 &
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run line_follower_perception line_follower_visualization' >/userdata/codex_visualization.log 2>&1 &
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run lanedetection obstacle' >/userdata/codex_obstacle.log 2>&1 &
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 run lanedetection lanedetection' >/userdata/codex_lanedetection.log 2>&1 &
sleep 3
nohup bash -lc 'source /opt/tros/humble/setup.bash && source /userdata/dev_ws/install/setup.bash && export ROS_DISABLE_LOANED_MESSAGES=1 && ros2 launch websocket websocket.launch.py websocket_only_show_image:=true websocket_image_topic:=/image_marked websocket_image_type:=mjpeg' >/userdata/codex_websocket_marked.log 2>&1 &
sleep 6
ps -ef | grep -E "originbot_base|hobot_usb_cam|vp100|hobot_codec|line_follower_perception|line_follower_visualization|lanedetection|obstacle_detection|websocket" | grep -v grep