#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

class YoloDNN {
public:
    cv::dnn::Net net;
    bool is_loaded = false;

    // COCO 数据集中车辆相关的 ID: 2(car), 5(bus), 7(truck)
    std::vector<int> target_classes = {2, 5, 7}; 

    // 初始化加载模型
    bool loadModel(const std::string& onnx_path) {
        try {
            net = cv::dnn::readNetFromONNX(onnx_path);
            // 强制使用 CPU 运行
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            is_loaded = true;
            std::cout << "\033[1;32m[YOLO C++] ONNX Model Loaded Successfully from: \033[0m" << onnx_path << std::endl;
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "\033[1;31m[YOLO C++ ERROR] Failed to load ONNX: \033[0m" << e.what() << std::endl;
            return false;
        }
    }

    // 执行推理并返回边框
    void detect(cv::Mat& img, std::vector<cv::Rect>& out_boxes) {
        out_boxes.clear();
        if (!is_loaded || img.empty()) return;

        // VINS 的图通常是灰度图，YOLO 需要 RGB 3通道
        cv::Mat rgb_img;
        if (img.channels() == 1) {
            cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2RGB);
        } else {
            rgb_img = img;
        }

        // 1. 预处理：缩放为 YOLO 要求的 640x640，归一化 1/255.0
        cv::Mat blob;
        cv::dnn::blobFromImage(rgb_img, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);

        // 2. 前向传播计算
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 3. 后处理解析 YOLOv8 的输出 [1, 84, 8400]
        cv::Mat out = outputs[0];
        int rows = out.size[2]; // 8400 个预测框
        int dimensions = out.size[1]; // 84 (4个坐标 + 80个类别)

        out = out.reshape(1, dimensions);
        cv::transpose(out, out); // 转置为 [8400, 84] 方便遍历

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        float x_factor = img.cols / 640.0;
        float y_factor = img.rows / 640.0;
        float* data = (float*)out.data;

        for (int i = 0; i < rows; ++i) {
            float* classes_scores = data + 4; // 从第5个元素开始是类别分数
            cv::Mat scores(1, 80, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // 过滤：置信度 > 0.4 且类别是车辆
            if (max_class_score > 0.4) {
                if (std::find(target_classes.begin(), target_classes.end(), class_id.x) != target_classes.end()) {
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];

                    // 还原到原始图像尺寸
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back((float)max_class_score);
                    class_ids.push_back(class_id.x);
                }
            }
            data += dimensions;
        }

        // 4. NMS (非极大值抑制)，去除重叠框
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.4, 0.45, indices);
        for (int idx : indices) {
            out_boxes.push_back(boxes[idx]);
        }
    }
};