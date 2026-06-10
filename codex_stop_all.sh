#!/bin/bash
set +e

publish_zero_cmd() {
  bash -lc '
    source /opt/tros/humble/setup.bash >/dev/null 2>&1 || true
    source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
    source /userdata/dev_ws/install/setup.bash >/dev/null 2>&1 || true
    export ROS_DISABLE_LOANED_MESSAGES=1
    timeout 2 ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
      -r 10 >/tmp/codex_stop.log 2>&1 || true
  '
}

kill_high_level_nodes() {
  pkill -f "line_follower_perception" || true
  pkill -f "line_follower_visualization" || true
  pkill -f "lanedetection" || true
  pkill -f "obstacle_detection" || true
  pkill -f "dnn_node_example" || true
  pkill -f "websocket.launch.py" || true
  pkill -f "/opt/tros/humble/lib/websocket/websocket" || true
  pkill -f "hobot_codec.launch.py" || true
  pkill -f "hobot_codec_decode.launch.py" || true
}

kill_bringup_nodes() {
  pkill -f "originbot.launch.py" || true
  pkill -f "vp100.launch.py" || true
  pkill -f "ros2 launch originbot_bringup originbot.launch.py" || true
  pkill -f "ros2 launch originbot_bringup vp100.launch.py" || true
  pkill -f "vp100_ros2_node" || true
  pkill -f "originbot_base" || true
  pkill -f "hobot_usb_cam" || true
  pkill -f "static_transform_publisher" || true
}

publish_zero_cmd
sleep 1
kill_high_level_nodes
sleep 1
publish_zero_cmd
sleep 1
kill_bringup_nodes
sleep 1

ps -eo pid,cmd \
  | grep -E "originbot_base|hobot_usb_cam|vp100_ros2_node|hobot_codec|line_follower_perception|line_follower_visualization|lanedetection|dnn_node_example|websocket|ros2 launch originbot_bringup|ros2 launch hobot_codec|ros2 launch websocket|static_transform_publisher" \
  | grep -v grep \
  || true
