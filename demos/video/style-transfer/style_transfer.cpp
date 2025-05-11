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

static torch::Device default_device_st = torch::Device(torch::kMPS);


torch::jit::Module load_model(const std::string& model_path) {
    std::cout << "Loading model from path: " << model_path << std::endl;
    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Moving model to device..." << std::endl;
        auto device = cvtool::get_default_device();
        module.to(device);
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

at::Tensor run_model(torch::jit::Module& module, const at::Tensor& input) {

    auto input_dtype = input.dtype();
    // std::cout.flush();
    // std::cout << "Input dtype: " << input.dtype() << std::endl;
    // std::cout << "Input sizes: " << input.sizes() << std::endl;
    // std::cout << "Input device: " << input.device() << std::endl;
    // std::cout.flush();
    // std::system("pause");

    // auto model_dtype = module.dtype();
    // std::cout << "Module: " << module << std::endl;


    // module.to(torch::kMPS);
    // module.eval();


    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // std::cout << "Input tensor: " << input.sizes() << std::endl;
    // auto output = module.forward(inputs).toTensor();
    auto output = module.forward(inputs).toTensor();

    // std::cout << "Model output: " << output.sizes() << std::endl;
    return output;
}

torch::Tensor eval_model(torch::jit::Module& module, const torch::Tensor& input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Forward pass
    auto output = module.forward(inputs).toTensor();

    return output;
}

torch::indexing::Slice slice() {
    return torch::indexing::Slice();
}

torch::Tensor test_channel(torch::Tensor& input) {
    std::cout << "Input device: " << input.device() << std::endl;

    int channel_to_disable = 0;
    // auto img = input.select(1, channel_to_disable).zero();  
    auto output = input.clone();
    output.select(1, channel_to_disable).zero_();
    // auto output = img;
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
        std::cout << "MPS is not available. Using CPU instead. " << std::endl;
    }
    cvtool::set_default_device(default_device_st);

    auto device = cvtool::get_default_device();

    // default_device = default_device_st;

    // std::string model_path = "style-transfer/models/mosaic_float32.pt";
    std::string model_path = "style-transfer/models/mosaic_float16.pt" ;
    torch::jit::Module module = load_model(model_path);
/*
    // module.to(torch::kFloat16);
    torch::Tensor input = torch::randn({1, 3, 1080, 1920}, device);
    std::cout << "Input tensor: " << input.sizes() << std::endl;
    std::cout << "Input tensor dtype: " << input.dtype() << std::endl;
    std::cout << "Input tensor device: " << input.device() << std::endl;
    // std::cout << "Model device: " << module.device() << std::endl;
    // std::cout << "Model dtype: " << module.dtype() << std::endl;

    torch::Tensor output = run_model(module, input);

    // Print the output tensor
    std::cout << "Output tensor: " << output.sizes() << std::endl;
*/
    return run_webcam_model(module, 0, 60, false, "");

}


int run_webcam_model(torch::jit::Module& module, int cam_index, int max_fps, bool is_video_loop, std::string vid_path = "") {

    torch::Device device = cvtool::get_default_device();

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

    auto camera_resolution = get_camera_resolution(cap);
    int height = std::get<0>(camera_resolution);
    int width  = std::get<1>(camera_resolution);


    cv::Mat frame_bgr;
    cv::Mat output_bgr;

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
        double delta_time = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
        auto fps = 1.0 / delta_time;
        std::cout << "\r[INFO] FPS: " << fps << " fps" << std::flush;
        double sleep_time = (1.0 / ((double)max_fps)) - delta_time;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));



        bool skip = true;
        if (skip) {

            auto start = std::chrono::high_resolution_clock::now();

            cv::Mat frame_rgb;
            cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

            auto input_tensor = to_tensor(frame_rgb,device);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Elapsed time (1): " << elapsed.count() * 1000.0 << " ms\n";


            // // // works
            start = std::chrono::high_resolution_clock::now();
            auto input = input_tensor.div_(255.0);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Elapsed time (2): " << elapsed.count() * 1000.0 << " ms\n";

            // auto input = input_tensor.to(device,true).to(torch::kFloat16) / 255.0;

            start = std::chrono::high_resolution_clock::now();
            auto model_output = run_model(module,input).div_(255.0);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Elapsed time (3): " << elapsed.count() * 1000.0 << " ms\n";


            start = std::chrono::high_resolution_clock::now();
            output_bgr = to_mat(model_output, cv::COLOR_RGB2BGR);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Elapsed time (4): " << elapsed.count() * 1000.0 << " ms\n";

            // // works
            // auto processed_input = prepped_input;
            // auto out_processed_input = processed_input.to(torch::kCPU,true);
            // frame_rgb = to_mat(out_processed_input);
            // cv::cvtColor(frame_rgb, output_bgr, cv::COLOR_RGB2BGR); 

            // works
            // auto out_mps_tensor = mps_tensor.to(torch::kCPU,true);
            // frame_rgb = to_mat(out_mps_tensor);
            // cv::cvtColor(frame_rgb, output_bgr, cv::COLOR_RGB2BGR); 



            // // works
            // frame_rgb = to_mat(input_tensor);
            // cv::cvtColor(frame_rgb, output_bgr, cv::COLOR_RGB2BGR); 



        } else {

            auto input_tensor = to_tensor(frame_bgr);

            auto mps_tensor = input_tensor.to(device,true);

            auto prepped_input = preprocess_input(mps_tensor);

            // Forward pass
            auto output = eval_model(module, prepped_input);
            auto processed_output = output.to(torch::kCPU,true);
            
            output_bgr = to_mat(processed_output);
        }

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