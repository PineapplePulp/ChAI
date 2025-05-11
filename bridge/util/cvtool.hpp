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

cv::VideoCapture open_camera(const std::string &file_path) {
    cv::VideoCapture cap(file_path);
    if (!cap.isOpened()) {
        std::cerr << "Could not open file " << file_path << std::endl;
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



// std::shared_ptr<at::Tensor> get_frame_buffer_tensor(int height,int width) {
//     auto options_cpu  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor frame_tensor_cpu = torch::empty({1, height, width, 3}, options_cpu);  
// }

std::shared_ptr<at::Tensor> create_buffer_tensor(
    torch::IntArrayRef sizes,
    torch::ScalarType = torch::kFloat32,
    torch::Device device = get_default_device()) {
    auto options_device = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(default_device);
    auto tensor = torch::empty(sizes, options_device);
    auto frame_tensor_device = std::make_shared<at::Tensor>(tensor);
    return frame_tensor_device;
}


std::shared_ptr<at::Tensor> create_frame_buffer_tensor(int height,int width,torch::Device device = get_default_device()) {
    torch::IntArrayRef sizes = {1, height, width, 3};
    return create_buffer_tensor(sizes, torch::kFloat32);
}

at::Tensor to_tensor(cv::Mat &img) {
    auto t = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kUInt8).clone();
    t = t.to(default_device);
    t = t.to(torch::kFloat32).permute({0, 3, 1, 2}) / 255.0;
    return t;//.to(default_device,true);
}

cv::Mat to_mat(at::Tensor &tensor) {
    // Ensure the tensor is on the CPU and not on the GPU
    // at::Tensor cpu_tensor = tensor.to(torch::kCPU);

    // Clone the tensor to avoid modifying the original data
    // at::Tensor cloned_tensor = cpu_tensor.clone();

    
    int height = tensor.size(2);
    int width = tensor.size(3);
    auto t = tensor
                .mul(255)
                .squeeze()
                .detach()
                .permute({1, 2, 0})
                .contiguous()
                .to(torch::kUInt8)
                // .clamp(0, 255)
                .clone()
                .to(torch::kCPU);
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
        // default_device = torch::Device(torch::kMPS);
        std::cout << "[INFO] Running on MPS" << std::endl;
    } else {
        std::cout << "[INFO] MPS not available, falling back to CPU" << std::endl;
    }
    return default_device;
}

// torch::jit::Module load_module_from_file(std::string model_path) {
//     std::string mp(reinterpret_cast<const char*>(model_path));

//     std::cout << "Loading model from path: " << mp << std::endl;
//     std::cout.flush();

//     torch::jit::Module module;
//     try
//     {
//         // Deserialize the ScriptModule from a file using torch::jit::load().
//         module = torch::jit::load(mp);
//     }
//     catch (const c10::Error& e)
//     {
//         std::cerr << "error loading the model\n" << e.msg();
//         std::system("pause");
//     }

//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(t_input);

//     return module;
// }

at::Tensor imagenet_resize(at::Tensor& image, int height, int width) {
    // Resize the image to the specified height and width
    auto resized_image = torch::nn::functional::interpolate(
        image,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({height, width}))
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    return resized_image;
}

at::Tensor imagenet_normalize_tensor(at::Tensor& input) {
    // Normalize the image using ImageNet mean and std
    // auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
    // auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
    // return (image - mean) / std;

    // std::cout << "Input sizes: " << input.sizes() << std::endl;

    at::Tensor image = input.to(torch::kFloat32).clone();// / 255.0;
    // std::cout << "Image sizes: " << image.sizes() << std::endl;

    static const std::vector<float> mean_data{0.485, 0.456, 0.406};
    static const std::vector<float> std_data{0.229, 0.224, 0.225};
    auto options = image.options();
    auto mean = torch::tensor(mean_data,options).reshape({3, 1, 1});  // (3,1,1)
    auto std  = torch::tensor(std_data,options).reshape({3, 1, 1});

    if (image.dim() == 4) {
        mean = mean.unsqueeze(0); // (1,3,1,1)
        std = std.unsqueeze(0);
    }

    // std::cout << "Mean sizes: " << mean.sizes() << std::endl;
    // std::cout << "Std sizes: " << std.sizes() << std::endl;
    // std::cout << "Image sizes: " << image.sizes() << std::endl;
    // std::exit(0);

    auto output = (image - mean) / std;
    output = output;
    // std::cout << "Output sizes: " << output.sizes() << std::endl;
    return output;
}


int show_webcam(int cam_index) {
    cv::VideoCapture cap = open_camera(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera index " << cam_index << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture image from camera" << std::endl;
            break;
        }

        cv::imshow("Webcam", frame);
        if (cv::waitKey(30) >= 0) break; // Exit on any key press
    }
    return 0;
}



at::Tensor capture_webcam(int cam_index) {
    cv::VideoCapture cap = open_camera(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera index " << cam_index << std::endl;
        return at::Tensor();
    }

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "Failed to capture image from camera" << std::endl;
        return at::Tensor();
    }

    auto tensor = to_tensor(frame);
    return tensor;
}
