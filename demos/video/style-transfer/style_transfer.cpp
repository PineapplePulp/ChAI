#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <utility>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cvtool.hpp>



int run_webcam_model(torch::jit::Module& module, int cam_index, int max_fps, bool is_video_loop, std::string vid_path);

static torch::Device default_device_st = torch::Device(torch::kCPU);


torch::jit::Module load_model(const std::string& model_path) {
    std::cout << "Loading model from path: " << model_path << std::endl;
    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Moving model to device..." << std::endl;
        module.to(default_device_st);
        std::cout << "Model moved to device." << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n" << e.msg();
    }
    std::cout << "Model loaded successfully." << std::endl;
    return module;

}

torch::Tensor preprocess_input(const torch::Tensor& input) {
    // Preprocess the input tensor as needed
    // For example, normalize the input tensor
    // auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
    // auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
    // return (input - mean) / std;
    return input;
}

torch::Tensor run_model(torch::jit::Module& module, const torch::Tensor& input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    std::cout << "Input tensor: " << input.sizes() << std::endl;
    auto output = module.forward(inputs).toTensor();
    std::cout << "Model output: " << output.sizes() << std::endl;
    return output;
}



int main() {
    // Load the model
    // std::string model_path = "style-transfer/models/my_module.pt";
    // torch::jit::Module module = load_model(model_path);
    // torch::Tensor input = torch::randn({10});
    // torch::Tensor output = run_model(module, input);


    if (torch::mps::is_available()) {
        default_device_st = torch::Device(torch::kMPS);
        std::cout << "MPS is available and set as the default device." << std::endl;
    } else {
        default_device_st = torch::Device(torch::kCPU);
        std::cout << "MPS is not available. Using CPU instead." << std::endl;
    }

    // default_device = default_device_st;

    std::string model_path = "style-transfer/models/mosaic.pt";
    torch::jit::Module module = load_model(model_path);
    torch::Tensor input = torch::randn({1, 3, 1428, 1904}, default_device_st);
    torch::Tensor output = run_model(module, input);

    // Print the output tensor
    std::cout << "Output tensor: " << output.sizes() << std::endl;

    return run_webcam_model(module, 0, 60, false, "");

}

int run_webcam_model(torch::jit::Module& module, int cam_index, int max_fps, bool is_video_loop, std::string vid_path = "") {

    torch::Device device = default_device_st;

    module.eval();
    module.to(device);

    bool video_loop = false;
    cv::VideoCapture cap;
    if (is_video_loop) { 
        cap = open_camera(vid_path);
        video_loop = true;
    } else {
        cap = open_camera(cam_index);
    }


    // 4. Pre‑allocate tensor to avoid dynamic allocations
    // int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    auto camera_resolution = get_camera_resolution(cap);
    int height = std::get<0>(camera_resolution);
    int width  = std::get<1>(camera_resolution);



    // // NHWC float32 frame buffer (1, H, W, 3)
    // auto options_cpu  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    // torch::Tensor frame_tensor_cpu  = torch::empty({1, height, width, 3}, options_cpu);

    // // MPS device tensor gets created lazily (to avoid copies when MPS unavailable)
    // torch::Tensor frame_tensor_device;
    // if (device.is_mps()) {
    //   frame_tensor_device = frame_tensor_cpu.to(device, /*non_blocking=*/true);
    // }

    // auto frame_tensor_device = create_frame_buffer_tensor(height, width, device);

    cv::Mat frame_bgr;
    cv::Mat output_bgr;
    // cv::Mat frame_rgb(height, width, CV_32FC3, frame_tensor_device->data_ptr());

    const auto to_mps = [&](torch::Tensor& t){ return device.is_mps() ? t.to(device, /*non_blocking=*/true) : t; };

    torch::NoGradGuard no_grad;                 // inference only

    std::chrono::time_point<std::chrono::system_clock> start_total = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> last_update = std::chrono::system_clock::now();

    size_t frame_count = 0;
    size_t last_frame_count = 0;

    while (true) {
        // std::cout << "\r[INFO] Processing frame... " << frame_count + 1 << std::flush;

        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
        if (video_loop && frame_count > 0) {
            cap = open_camera(vid_path);
            frame_count = 0;
            last_frame_count = 0;
            start_total = std::chrono::system_clock::now();
            last_update = std::chrono::system_clock::now(); // ??? not sure
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            std::cout << "[INFO] Replaying video..." << std::endl;
            continue;
        }
        std::cerr << "[WARN] Empty frame, exiting" << std::endl;
        break;
        }


        ++frame_count;
        const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
        auto delta = now - last_update;
        // std::chrono::milliseconds delta_millis = std::chrono::duration_cast<std::chrono::microseconds>(delta);
        double delta_time = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
        auto fps = 1.0 / delta_time;
        std::cout << "\r[INFO] FPS: " << fps << " fps" << std::flush;

        // Display (optional)

        double sleep_time = (1.0 / ((double)max_fps)) - delta_time;
        
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));


        auto input_tensor = to_tensor(frame_bgr).clone();
        auto mps_tensor = input_tensor.to(torch::kMPS,true).clone();

        auto prepped_input = preprocess_input(mps_tensor);

        // Forward pass
        auto output = run_model(module, prepped_input).clone();
        auto processed_output = output.to(torch::kCPU,true).clone();

        output_bgr = to_mat(processed_output);

        cv::imshow("webcam", output_bgr);

        // Display FPS

        // std::cout << "[INFO] dt: " << delta_time << std::endl;
        // std::cout << "[INFO] FPS: " << fps << std::endl;




        // std::thread::sleep_for(std::chrono::milliseconds(700));

        // std::thread::sleep_for()


        // std::thread::sleep_for(std::chrono::milliseconds(expected_time_index - (delta_time + last_time_index)));


        last_frame_count = frame_count;
        last_update = now; // std::chrono::system_clock::now();
        if (cv::waitKey(1) == 27) { // ESC key
        break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}