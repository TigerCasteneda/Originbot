// Copyright (c) 2024, D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LINE_FOLLOWER_VISUALIZATION_H_
#define LINE_FOLLOWER_VISUALIZATION_H_

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/bool.hpp>

#include <opencv2/opencv.hpp>

using rclcpp::NodeOptions;

class LineFollowerVisualizationNode : public rclcpp::Node {
 public:
  LineFollowerVisualizationNode(const std::string &node_name,
                                const NodeOptions &options = NodeOptions());
  ~LineFollowerVisualizationNode();

 private:
  void image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void point_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
  void cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void obs_callback(const std_msgs::msg::Bool::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr point_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr obs_sub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_pub_;

  cv::Point latest_point_{-1, -1};
  bool has_point_ = false;
  double latest_linear_x_ = 0.0;
  double latest_angular_z_ = 0.0;
  bool has_obstacle_ = false;
};

#endif  // LINE_FOLLOWER_VISUALIZATION_H_
