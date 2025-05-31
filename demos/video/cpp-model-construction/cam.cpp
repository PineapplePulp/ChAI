#include "transformer_net.hpp"


#include <opencv2/opencv.hpp>
#include <iostream>


TransformerNet load_net() {
    torch::manual_seed(0);
  std::cout << "set seed\n";
  TransformerNet model;
  std::cout << "TransformerNet model created.\n";

  model->load_parameters("/Users/iainmoncrief/Documents/Github/ChAI/demos/video/cpp-model-construction/state_dict_raw.pt");
  // torch::serialize::InputArchive archive;
  // archive.load_from("/Users/iainmoncrief/Documents/Github/ChAI/demos/video/cpp-model-construction/incomplete_sunday_afternoon.model");   // load the raw weights :contentReference[oaicite:2]{index=2}
  // std::cout << "Loading model from archive...\n";
  // model->load(archive);
  // std::cout << "Model loaded successfully!\n";
  // model->eval();
  std::cout << "Model is in evaluation mode.\n";

  // dummy input
  auto input = torch::randn({1,3,256,256});
  std::cout << "Input shape: " << input.sizes() << "\n";
  auto output = model->forward(input);

  std::cout << "Output shape: " << output.sizes() << "\n";

  return model;
}


cv::Mat new_frame(cv::Mat &frame,TransformerNet &model) {

    cv::Mat rgb_float_frame;
    cv::cvtColor(frame, rgb_float_frame, cv::COLOR_BGR2RGB);
    rgb_float_frame.convertTo(rgb_float_frame, CV_32FC3, 1.0f/255.0f);

    // cv::MatSize size = rgb_frame.size;
    // std::cout << "x " << size[0] << " y " << size[1] << " channels " << rgb_frame.dims << std::endl;
    int64_t height = rgb_float_frame.rows;
    int64_t width = rgb_float_frame.cols;
    int64_t channels = rgb_float_frame.channels();
    int64_t pixels = rgb_float_frame.total();
    int64_t size = pixels * channels;

    // std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << ", Size: " << size << std::endl;

    torch::Tensor tensor = torch::from_blob(rgb_float_frame.data, 
                                            {height,width, channels}, 
                                            torch::kFloat32).to(torch::kCPU, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);

    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).contiguous();

    torch::Tensor output_tensor = model->forward(tensor);
    output_tensor = output_tensor.squeeze(0).permute({1, 2, 0}).contiguous();
    output_tensor = output_tensor.to(torch::kCPU, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);
    output_tensor.div_(255.0);

    // chpl_external_array 
    //     rgb_float_frame_data_ptr = chpl_make_external_array_ptr(rgb_float_frame.data,size);
    
    // chpl_external_array 
    //     rgb_float_output_frame_array = getNewFrame(&rgb_float_frame_data_ptr, height, width, channels);
    

    // cv::Mat new_rgb_frame(height, width, CV_8UC3,new_frame_array.elts);
    // cv::cvtColor(new_rgb_frame, new_rgb_frame, cv::COLOR_RGB2BGR);

    cv::Mat output_frame(height,width,CV_32FC3,output_tensor.data_ptr<float>()); // frame to write to
    output_frame.convertTo(output_frame, CV_8UC3, 255.0f); 
    cv::cvtColor(output_frame, output_frame, cv::COLOR_RGB2BGR);

    return output_frame;


}


int mirror() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the webcam.\n";
        return -1;
    }

    TransformerNet model = load_net();

    cv::Mat frame;
    const std::string windowName = "Webcam Feed";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Size original_frame_size;
    cv::Size processed_frame_size;

    while (true) {

        uint64_t start = cv::getTickCount();

        // Capture a new frame from webcam
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured.\n";
            break;
        }
        
        original_frame_size = frame.size();

        const int width = (original_frame_size.width * 0.2);
        const int height = (original_frame_size.height * 0.2);
        processed_frame_size = cv::Size(width, height);
        cv::resize(frame, frame, processed_frame_size);

        // std::cout << "Frame size: " << frame.size() << std::endl;
        // std::cout << "New frame size: " << processed_frame_size << std::endl;

        cv::Mat next_frame = new_frame(frame,model);

        cv::resize(next_frame, next_frame, original_frame_size);

        // Display the captured frame
        cv::imshow(windowName, next_frame);

        // Wait for 30ms or until 'q' key is pressed
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "\rcv::FPS : " << fps << "\t\r" << std::flush;
    }

    // Release the camera and destroy all windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


int main(int argc, char* argv[]) {
    return mirror();
}


