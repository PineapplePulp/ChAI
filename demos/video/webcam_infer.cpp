#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <utility>

#include <cvtool.hpp>
#include "imageops.hpp"


torch::Tensor sobel_dx = torch::tensor({{-1, 0, 1},
                                    {-2, 0, 2},
                                    {-1, 0, 1}}).to(torch::kFloat32);
torch::Tensor sobel_dy = torch::tensor({{-1, -2, -1},
                                    {0, 0, 0},
                                    {1, 2, 1}}).to(torch::kFloat32);

torch::Tensor sobel_kernel = torch::cat({sobel_dx, sobel_dy}, 0).unsqueeze(0).unsqueeze(0);

torch::Tensor sobel_conv(torch::Tensor& input) {
    // Convert input to float32
    // input = input.to(torch::kFloat32);

    // // Apply Sobel filter
    // auto sobel_x = at::conv2d(input, sobel_kernel, /*bias=*/{}, /*stride=*/1, /*padding=*/1);
    // auto sobel_y = at::conv2d(input, sobel_kernel.transpose(0, 1), /*bias=*/{}, /*stride=*/1, /*padding=*/1);

    // // Compute magnitude
    // auto sobel_magnitude = torch::sqrt(sobel_x.pow(2) + sobel_y.pow(2));


    auto output = torch::conv2d(input, sobel_kernel, {},1,1);

    return output;
}

struct Model : torch::nn::Module {
  Model() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 10));
    r = register_parameter("r", torch::rand({1, 3, 224, 224}));
    uninitialized = true;

    std::cout << "Sobel kernel: " << sobel_kernel.sizes() << std::endl;
  }

  torch::Tensor forward(torch::Tensor x) {
    if (uninitialized) {
      // Initialize the tensor with random values
      r = torch::rand(x.sizes(), torch::kFloat32).to(x.device());
      uninitialized = false;
      std::cout << "Input sizes: " << x.sizes() << std::endl;
    }
    // auto output = x + r;
    auto input = x;
    // auto output = imageops::sobel_rgb(input);
    auto output = imagenet_normalize_tensor(input);
    return output;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::Tensor r ;
  bool uninitialized;
};


int max_fps = 60;

int main(int argc, char** argv) {

  show_webcam(0);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path/to/torchscript_model.pt> [cam_index]" << std::endl;
    return -1;
  }

  std::string model_path = argv[1];
  int cam_index = (argc > 2) ? std::stoi(argv[2]) : 0;

  // 1. Select device (MPS if available, else CPU)
  // torch::Device device(torch::kCPU);
// #ifdef TORCH_MPS_AVAILABLE
  // if (torch::mps::is_available()) {
  //   device = torch::Device(torch::kMPS);
  //   std::cout << "[INFO] Running on MPS" << std::endl;
  // } else {
  //   std::cout << "[INFO] MPS not available, falling back to CPU" << std::endl;
  // }
// #else
//   std::cout << "[INFO] Built without TORCH_MPS_AVAILABLE, running on CPU" << std::endl;
// #endif
    // {
    //   torch::Device device(torch::kMPS);

    //   torch::Tensor a = torch::rand({2, 3}, device);
    //   std::cout << a << std::endl;
    //   std::cout << "Tensor: " << a.device() << std::endl;
    // }

  // 2. Load TorchScript module
//   torch::jit::script::Module module;
//   try {
//     module = torch::jit::load(model_path, device);
//   } catch (const c10::Error& e) {
//     std::cerr << "Error loading the model: " << e.what() << std::endl;
//     return -1;
//   }
//   module.eval();

//   auto model = std::make_shared<Model>();
//   model->eval();


  torch::Device device = get_default_device();

  Model module;
  module.eval();
  module.to(device);

  // std::string module_path = argv[1];
  // std::cout << "Loading model from path: " << module_path << std::endl;
  // torch::jit::Module module = load_module_from_file(model_path);
  // std::cout << "Model loaded successfully" << std::endl;


  // 3. Setup webcam
  // cv::VideoCapture cap(cam_index, cv::CAP_AVFOUNDATION);
  // if (!cap.isOpened()) {
  //   std::cerr << "Could not open camera index " << cam_index << std::endl;
  //   return -1;
  // }
  // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);        // minimal internal buffering
  // cap.set(cv::CAP_PROP_FPS, 60);              // request higher FPS if possible

  bool video_loop = false;
  std::string vid_path;
  cv::VideoCapture cap;
  if (argc > 3) { 
    vid_path = argv[3];
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




    auto input_tensor = to_tensor(frame_bgr);
    // auto input_tensor = frame_tensor_device->permute({0, 3, 1, 2});

    // std::cout << "input_tensor: " << input_tensor.sizes() << std::endl;
    // std::cout << "input_tensor max: " << torch::max(input_tensor) << std::endl;
    // input_tensor = to_mps(input_tensor);

    // Forward pass
    auto output = module.forward(input_tensor);
    // std::cout << "\r[INFO] Inference time: " << output.sizes() << std::endl;
    // auto output_bgr = to_mat(output);
    output_bgr = to_mat(output);

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
