#pragma once

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <utility>


namespace cvtool {
    static torch::Device default_device(torch::kCPU);
    static bool default_device_set = false;
    static torch::Device set_default_device(torch::Device device) {
        default_device = device;
        default_device_set = true;
        return default_device;
    }
    torch::Device get_default_device() {
        if (!default_device_set) {
            if (torch::mps::is_available()) {
                std::cout << "[INFO] Running on MPS" << std::endl;
                default_device = torch::Device(torch::kMPS);
            } else {
                std::cout << "[INFO] MPS not available, falling back to CPU" << std::endl;
                default_device = torch::Device(torch::kCPU);
            }
        }
        return default_device;
    }

    bool can_get_default_device() {
        return default_device_set || !torch::mps::is_available();
    }

    torch::Device get_host_device() {
        return torch::Device(torch::kCPU);
    }
}

// enum CVToColorPermutation {
//     RGB_TO_BGR = cv::COLOR_RGB2BGR,
//     BGR_TO_RGB = cv::COLOR_BGR2RGB,
// };

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

at::Tensor to_tensor(cv::Mat &frame, torch::Device device = default_device) {


    auto t = at::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kUInt8).permute({0, 3, 1, 2}).clone();
    auto options = at::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(device)
                    .requires_grad(false);
    return t.to(options,true).contiguous().div_(255.0);

    // t = t.to(default_device,);
    // t = t.to(torch::kFloat32).permute({0, 3, 1, 2}).contiguous() / 255.0;

    // return t;//.to(default_device,true);
}

// at::Tensor to_tensor(cv::Mat &img, cv::ColorConversionCodes color_conversion = cv::COLOR_BGR2RGB) {
//     auto t = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kUInt8).clone();
//     t = t.to(default_device);
//     t = t.to(torch::kFloat32).permute({0, 3, 1, 2}) / 255.0;
//     return t;//.to(default_device,true);
// }

// at::Tensor to_tensor(cv::Mat &img, cv::ColorConversionCodes color_conversion = cv::COLOR_BGR2RGB, device = ) {
//     auto t = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kUInt8).clone();
//     t = t.to(default_device);
//     t = t.to(torch::kFloat32).permute({0, 3, 1, 2}) / 255.0;
//     return t;//.to(default_device,true);
// }

// at::Tensor to_tensor(cv::Mat &img, torch::Device device = cvtool::get_default_device()) {
//     auto img_t = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kUInt8);
//     auto t = img_t.clone().to(device);
//     t = t.to(torch::kFloat32).permute({0, 3, 1, 2}) / 255.0;
//     return t;//.to(default_device,true);
// }

//--------------------------------------------------------------------
// • img : any H×W×C OpenCV matrix (CV_8U, CV_32F, CV_16F …, planar or packed)
// • device : torch::kCUDA, torch::kMPS or torch::kCPU (default = current CUDA if available)
//--------------------------------------------------------------------
at::Tensor to_tensor_(const cv::Mat& img, torch::Device device = get_default_device())
{
    // 1. Make sure the source data are contiguous
    cv::Mat contiguous = img.isContinuous() ? img : img.clone();

    // 2. Convert pixel type to 32‑bit float in [0,1] so we keep enough
    //    head‑room for the later FP16 cast.  (OpenCV has only limited
    //    native FP16 support, so converting to CV_32F first is usually
    //    safer and portable.)
    cv::Mat float32;
    contiguous.convertTo(float32, CV_32F, 1.0 / 255.0);   // scale if img was CV_8U

    // 3. Wrap the OpenCV buffer with a *view* tensor (no copy yet).
    auto tmp = torch::from_blob(
                  float32.data,                             // raw pointer
                  {float32.rows, float32.cols, float32.channels()},
                  torch::TensorOptions().dtype(torch::kFloat32));

    // 4. Re‑arrange to CHW, move to wanted device, cast to FP16 *and* copy
    //    so that the returned tensor owns its storage (clone() is mandatory).
    auto t = tmp.permute({2, 0, 1})                        // HWC → CHW
                 .to(device, /*dtype=*/torch::kFloat16,
                     /*non_blocking=*/true, /*copy=*/true) // copy = true ⇒ owns memory
                 .clone();                                 // guarantees ownership

    return t; //  C×H×W, float16, on CUDA / MPS / CPU
}


cv::Mat to_mat(at::Tensor &tensor) {
    // Ensure the tensor is on the CPU and not on the GPU
    // at::Tensor cpu_tensor = tensor.to(torch::kCPU);

    // Clone the tensor to avoid modifying the original data
    // at::Tensor cloned_tensor = cpu_tensor.clone();

    
    int height = tensor.size(2);
    int width = tensor.size(3);
    auto t = tensor
                .detach()
                .squeeze()
                .contiguous()
                .mul(255.0)
                .clamp(0, 255)
                .permute({1, 2, 0})
                .contiguous()
                .to(torch::kUInt8)
                .clone()
                .to(at::kCPU,true);
                

    // auto t = tensor
    //             .mul(255)
    //             .squeeze()
    //             .detach()
    //             .permute({1, 2, 0})
    //             .contiguous()
    //             .to(torch::kUInt8)
    //             // .clamp(0, 255)
    //             .clone()
    //             // .to(cvtool::get_default_device(), /*non_blocking=*/true, /*copy=*/true)
    //             .to(torch::kCPU);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, t.data_ptr());
    return mat.clone();



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


cv::Mat to_mat(at::Tensor &tensor, cv::ColorConversionCodes color_conversion) {

    int height = tensor.size(2);
    int width = tensor.size(3);
    auto t = tensor
                // .to(torch::kFloat32)
                .mul(255.0)
                .clamp(0.0, 255.0)
                .to(torch::kUInt8)
                .squeeze()
                .detach()
                .permute({1, 2, 0})
                .contiguous()
                .clone()
                .to(torch::kCPU);
    cv::Mat mat = cv::Mat(height, width, CV_8UC3, t.data_ptr());
    cv::Mat mat2;
    cv::cvtColor(mat, mat2, color_conversion);
    return mat2.clone();
}

torch::Device get_default_device() {
    if (torch::mps::is_available()) {
        std::cout << "[INFO] Running on MPS" << std::endl;
        default_device = torch::Device(torch::kMPS);
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


torch::Tensor sobel_edge_detection(torch::Tensor& input,torch::Device device = cvtool::get_default_device()) {
    // // // Sobel edge detection
    // auto sobel_x = torch::tensor({{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}, input.dtype()).view({1, 1, 3, 3});
    // auto sobel_y = torch::tensor({{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}, input.dtype()).view({1, 1, 3, 3});
    // sobel_x.to(input.device());
    // sobel_y.to(input.device());

    // auto edges_x = torch::nn::functional::conv2d(input.unsqueeze(0), sobel_x);
    // auto edges_y = torch::nn::functional::conv2d(input.unsqueeze(0), sobel_y);

    // return (edges_x + edges_y).squeeze(0);


    torch::Tensor sobel_dx = torch::tensor({{-1, 0, 1},
                                            {-2, 0, 2},
                                            {-1, 0, 1}}).to(input.dtype());
    torch::Tensor sobel_dy = torch::tensor({{-1, -2, -1},
                                            {0, 0, 0},
                                            {1, 2, 1}}).to(input.dtype());
    sobel_dx.to(input.device());
    sobel_dy.to(input.device());


    torch::Tensor sobel_kernel = torch::cat({sobel_dx, sobel_dy}, 0).unsqueeze(0).unsqueeze(0);
    sobel_kernel.to(input.device());

    return torch::conv2d(input, sobel_kernel, {}, 1, 1);
}




