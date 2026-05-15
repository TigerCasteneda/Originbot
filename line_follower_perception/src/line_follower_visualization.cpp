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

#include "line_follower_perception/line_follower_visualization.h"

#include <algorithm>
#include <string>
#include <vector>

LineFollowerVisualizationNode::LineFollowerVisualizationNode(
    const std::string &node_name, const NodeOptions &options)
    : Node(node_name, options) {
  image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "image", rclcpp::SensorDataQoS(),
      std::bind(&LineFollowerVisualizationNode::image_callback, this,
                std::placeholders::_1));

  point_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "line_track_center_detection", 10,
      std::bind(&LineFollowerVisualizationNode::point_callback, this,
                std::placeholders::_1));
  cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&LineFollowerVisualizationNode::cmd_callback, this,
                std::placeholders::_1));
  obs_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/has_obs", 10,
      std::bind(&LineFollowerVisualizationNode::obs_callback, this,
                std::placeholders::_1));

  image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "image_marked", 10);

  RCLCPP_INFO(this->get_logger(),
              "LineFollowerVisualizationNode started: /image + "
              "/line_track_center_detection -> /image_marked");
}

LineFollowerVisualizationNode::~LineFollowerVisualizationNode() {}

void LineFollowerVisualizationNode::point_callback(
    const geometry_msgs::msg::PointStamped::SharedPtr msg) {
  latest_point_.x = static_cast<int>(msg->point.x);
  latest_point_.y = static_cast<int>(msg->point.y);
  has_point_ = latest_point_.x >= 0 && latest_point_.y >= 0;
}

void LineFollowerVisualizationNode::cmd_callback(
    const geometry_msgs::msg::Twist::SharedPtr msg) {
  latest_linear_x_ = msg->linear.x;
  latest_angular_z_ = msg->angular.z;
}

void LineFollowerVisualizationNode::obs_callback(
    const std_msgs::msg::Bool::SharedPtr msg) {
  has_obstacle_ = msg->data;
}

void LineFollowerVisualizationNode::image_callback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
  if (msg->data.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty compressed frame");
    return;
  }

  cv::Mat encoded(1, static_cast<int>(msg->data.size()), CV_8UC1,
                  const_cast<unsigned char *>(msg->data.data()));
  cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (image.empty()) {
    RCLCPP_WARN(this->get_logger(), "Failed to decode /image compressed frame");
    return;
  }

  const int image_width = image.cols;
  const int image_height = image.rows;
  const int crop_top = std::min(50, image_height - 1);
  const int crop_bottom = std::min(498, image_height - 1);
  const int image_center_x = image_width / 2;

  cv::line(image, cv::Point(0, crop_top), cv::Point(image_width - 1, crop_top),
           cv::Scalar(255, 255, 0), 2);
  cv::line(image, cv::Point(0, crop_bottom),
           cv::Point(image_width - 1, crop_bottom), cv::Scalar(255, 255, 0), 2);
  cv::line(image, cv::Point(image_center_x, 0),
           cv::Point(image_center_x, image_height - 1), cv::Scalar(255, 0, 0), 2);

  if (has_point_) {
    const int x = std::clamp(latest_point_.x, 0, image_width - 1);
    const int y = std::clamp(latest_point_.y, 0, image_height - 1);
    const int error = x - image_center_x;

    cv::circle(image, cv::Point(x, y), 16, cv::Scalar(0, 255, 0), -1);
    cv::line(image, cv::Point(x - 35, y), cv::Point(x + 35, y),
             cv::Scalar(0, 255, 0), 3);
    cv::line(image, cv::Point(x, y - 35), cv::Point(x, y + 35),
             cv::Scalar(0, 255, 0), 3);
    cv::line(image, cv::Point(image_center_x, y), cv::Point(x, y),
             cv::Scalar(0, 0, 255), 3);
    cv::putText(image,
                "track center x=" + std::to_string(x) +
                    " y=" + std::to_string(y) +
                    " err=" + std::to_string(error),
                cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                cv::Scalar(0, 255, 0), 3);
  } else {
    cv::putText(image, "waiting for inference point", cv::Point(30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
  }

  const std::string cmd_text =
      "cmd v=" + std::to_string(latest_linear_x_).substr(0, 5) +
      " w=" + std::to_string(latest_angular_z_).substr(0, 5);
  const std::string obs_text =
      std::string("obstacle=") + (has_obstacle_ ? "true" : "false");
  cv::putText(image, cmd_text, cv::Point(30, 110), cv::FONT_HERSHEY_SIMPLEX,
              1.0, cv::Scalar(255, 255, 255), 2);
  cv::putText(image, obs_text, cv::Point(30, 150), cv::FONT_HERSHEY_SIMPLEX,
              1.0, has_obstacle_ ? cv::Scalar(0, 0, 255)
                                 : cv::Scalar(0, 255, 0),
              2);

  std::vector<unsigned char> output_buffer;
  cv::imencode(".jpg", image, output_buffer, {cv::IMWRITE_JPEG_QUALITY, 85});

  sensor_msgs::msg::CompressedImage output_msg;
  output_msg.header = msg->header;
  output_msg.format = "jpeg";
  output_msg.data.assign(output_buffer.begin(), output_buffer.end());
  image_pub_->publish(output_msg);
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineFollowerVisualizationNode>(
      "line_follower_visualization"));
  rclcpp::shutdown();
  return 0;
}
