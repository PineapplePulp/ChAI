#pragma once

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <utility>


static torch::Device default_device(torch::kCPU);
torch::Device get_default_device();

cv::VideoCapture open_camera(int cam_index) {
    cv::VideoCapture cap(cam_index, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera index " << cam_index << std::endl;
        return cv::VideoCapture();
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // minimal internal buffering
    cap.set(cv::CAP_PROP_FPS, 60);       // request higher FPS if possible
    return cap;
}

std::pair<int,int> get_camera_resolution(cv::VideoCapture& cap) {
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return {height, width};
}

std::shared_ptr<cv::Mat> create_frame_buffer(int height, int width) {
    auto frame_buffer = std::make_shared<cv::Mat>(height, width, CV_8UC3);
    return frame_buffer;
}



// std::shared_ptr<torch::Tensor> get_frame_buffer_tensor(int height,int width) {
//     auto options_cpu  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor frame_tensor_cpu = torch::empty({1, height, width, 3}, options_cpu);  
// }

std::shared_ptr<torch::Tensor> create_buffer_tensor(
    torch::IntArrayRef sizes,
    torch::ScalarType = torch::kFloat32,
    torch::Device device = get_default_device()) {
    auto options_device = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(default_device);
    auto tensor = torch::empty(sizes, options_device);
    auto frame_tensor_device = std::make_shared<torch::Tensor>(tensor);
    return frame_tensor_device;
}


std::shared_ptr<torch::Tensor> create_frame_buffer_tensor(int height,int width,torch::Device device = get_default_device()) {
    torch::IntArrayRef sizes = {1, height, width, 3};
    return create_buffer_tensor(sizes, torch::kFloat32);
}

torch::Tensor to_tensor(cv::Mat &img) {
    auto t = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kUInt8);
    t = t.clone();
    // t = t.to(default_device);
    t = t.to(torch::kFloat32).permute({0, 3, 1, 2}) / 255.0;
    return t;//.to(default_device,true);
}

cv::Mat to_mat(torch::Tensor &tensor) {
    // Ensure the tensor is on the CPU and not on the GPU
    // torch::Tensor cpu_tensor = tensor.to(torch::kCPU);

    // Clone the tensor to avoid modifying the original data
    // torch::Tensor cloned_tensor = cpu_tensor.clone();

    
    int height = tensor.size(2);
    int width = tensor.size(3);
    auto t = tensor.squeeze()
                .detach()
                .permute({1, 2, 0})
                .contiguous()
                .mul(255)
                .clamp(0, 255)
                .to(torch::kUInt8)
                .clone()
                .to(torch::kCPU,true);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, t.data_ptr());
    return mat;



    // tensor = tensor.squeeze().detach();
    // tensor = tensor.permute({1, 2, 0}).contiguous();
    // tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    // tensor = tensor.to(torch::kCPU);
    // int64_t height = tensor.size(0);
    // int64_t width = tensor.size(1);
    // cv::Mat mat =
    //     cv::Mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr<uchar>());
    // return mat.clone();
}

torch::Device get_default_device() {
    if (torch::mps::is_available()) {
        default_device = torch::Device(torch::kMPS);
        std::cout << "[INFO] Running on MPS" << std::endl;
    } else {
        std::cout << "[INFO] MPS not available, falling back to CPU" << std::endl;
    }
    return default_device;
}