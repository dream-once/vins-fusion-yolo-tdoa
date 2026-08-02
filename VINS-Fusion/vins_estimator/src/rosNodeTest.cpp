/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
//new_add
#include <geometry_msgs/PointStamped.h>
// 在 #include <geometry_msgs/PointStamped.h> 下方添加
#include <yolo_vins_msgs/BoundingBoxes.h>

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;
// 加上 extern，告诉编译器去底层库里找它们！
extern std::vector<cv::Rect> yolo_boxes;
extern std::mutex yolo_mutex;
// [新增] YOLO 语义框缓冲队列
queue<yolo_vins_msgs::BoundingBoxesConstPtr> yolo_buf;

// ==== TDOA 魔改专区 ====
void tdoa_callback(const geometry_msgs::PointStampedConstPtr &msg)
{
    double t = msg->header.stamp.toSec();
    // 提取完整的三维距离差向量
    Eigen::Vector3d tdoa_vector(msg->point.x, msg->point.y, msg->point.z); 
    // 将向量传入大脑
    estimator.inputTDOA(t, tdoa_vector);
}
// =======================

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

// [新增] YOLO 话题回调
void yolo_callback(const yolo_vins_msgs::BoundingBoxesConstPtr &yolo_msg)
{
    m_buf.lock();
    yolo_buf.push(yolo_msg);
    m_buf.unlock();

    // 2. 更新全局共享框（给前端提取特征点做掩码用）
    std::lock_guard<std::mutex> lock(yolo_mutex);
    yolo_boxes.clear();
    for (const auto &box : yolo_msg->bounding_boxes)
    {
        int x = std::max(0, (int)box.xmin);
        int y = std::max(0, (int)box.ymin);
        int w = std::max(0, (int)(box.xmax - box.xmin));
        int h = std::max(0, (int)(box.ymax - box.ymin));
        yolo_boxes.push_back(cv::Rect(x, y, w, h));
    }
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            // [新增] 初始化空指针，用于防范 YOLO 延迟掉帧
            yolo_vins_msgs::BoundingBoxesConstPtr yolo_data = nullptr;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();

                    // [核心注入] 语义概率流与视觉流的硬对齐
                    // 清除过期的 YOLO 语义帧
                    while (!yolo_buf.empty() && yolo_buf.front()->header.stamp.toSec() < time - 0.003)
                    {
                        yolo_buf.pop();
                    }
                    // 匹配成功则提取，失败则保持 yolo_data 为 nullptr (系统退化为纯几何 VIO)
                    if (!yolo_buf.empty() && std::abs(yolo_buf.front()->header.stamp.toSec() - time) <= 0.003)
                    {
                        yolo_data = yolo_buf.front();
                        yolo_buf.pop();
                    }
                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if(!image0.empty())
                estimator.inputImage(time, image0, image1, yolo_data);
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            yolo_vins_msgs::BoundingBoxesConstPtr yolo_data = nullptr; // [新增] 初始化为空
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                // [新增] 语义时间戳对齐逻辑
                // 清理掉所有比当前图像时间戳老旧的 YOLO 帧 (容差 0.003s)
                while (!yolo_buf.empty() && yolo_buf.front()->header.stamp.toSec() < time - 0.003)
                {
                    yolo_buf.pop();
                }
                // 检查队首的 YOLO 帧是否与当前图像时间戳对齐
                if (!yolo_buf.empty() && std::abs(yolo_buf.front()->header.stamp.toSec() - time) <= 0.003)
                {
                    yolo_data = yolo_buf.front();
                    yolo_buf.pop();
                }
            }
            m_buf.unlock();
            if(!image.empty())
               estimator.inputImage(time, image, cv::Mat(), yolo_data);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        // 注意：如果你不接收外部 YOLO 概率，默认为 0 (静态背景)
        double p_d = 0.0;
       Eigen::Matrix<double, 8, 1> xyz_uv_velocity_prob;
        xyz_uv_velocity_prob << x, y, z, p_u, p_v, velocity_x, velocity_y, p_d;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity_prob);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    // 监听名为 "/uwb/tdoa" 的话题
    ros::Subscriber sub_tdoa = n.subscribe("/uwb/tdoa", 1000, tdoa_callback);

    // 在 sub_tdoa 附近添加
    ros::Subscriber sub_yolo = n.subscribe("/yolo/bounding_boxes", 100, yolo_callback);
    
    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
