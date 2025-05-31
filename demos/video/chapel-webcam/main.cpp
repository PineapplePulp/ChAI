#include "smol_wrapper.h"


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstddef>
#include <tuple>

#include <thread>
#include <chrono>


std::tuple<cv::Mat,std::size_t> new_frame(cv::Mat &frame) {

    cv::Mat rgb_float_frame;
    cv::cvtColor(frame, rgb_float_frame, cv::COLOR_BGR2RGB);
    rgb_float_frame.convertTo(rgb_float_frame, CV_32FC3, 1.0f/255.0f);

    int64_t height = rgb_float_frame.rows;
    int64_t width = rgb_float_frame.cols;
    int64_t channels = rgb_float_frame.channels();
    int64_t pixels = rgb_float_frame.total();
    int64_t size = pixels * channels;

    chpl_external_array 
        rgb_float_frame_data_ptr = chpl_make_external_array_ptr(rgb_float_frame.data,size);
    
    std::size_t chpl_start = cv::getTickCount();

    chpl_external_array 
        rgb_float_output_frame_array = getNewFrame(&rgb_float_frame_data_ptr, height, width, channels);
    
    std::size_t chpl_end = cv::getTickCount();
    
    chpl_free_external_array(rgb_float_frame_data_ptr);


    cv::Mat output_frame(height,width,CV_32FC3,rgb_float_output_frame_array.elts); // frame to write to
    output_frame.convertTo(output_frame, CV_8UC3, 255.0f); 
    cv::cvtColor(output_frame, output_frame, cv::COLOR_RGB2BGR);

    chpl_free_external_array(rgb_float_output_frame_array);

    std::tuple<cv::Mat,std::size_t> ouput_package = {output_frame,chpl_end - chpl_start};
    return ouput_package;

}


int mirror() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the webcam.\n";
        return -1;
    }

    cv::Mat frame;
    const std::string windowName = "Webcam Feed";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Size original_frame_size;
    cv::Size processed_frame_size;

    double last_frame_fps = 0.0;
    double last_chpl_fps = 0.0;

    while (true) {

        std::size_t frame_start = cv::getTickCount();

        // Capture a new frame from webcam
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured.\n";
            break;
        }
        
        original_frame_size = frame.size();

        const auto width = getScaledFrameWidth(original_frame_size.width);
        const auto height = getScaledFrameHeight(original_frame_size.height);
        processed_frame_size = cv::Size(width, height);
        cv::resize(frame, frame, processed_frame_size);

        // std::cout << "Frame size: " << frame.size() << std::endl;
        // std::cout << "New frame size: " << processed_frame_size << std::endl;

        cv::Mat next_frame;
        std::size_t chpl_delta;
        std::tie(next_frame,chpl_delta) = new_frame(frame);

        cv::resize(next_frame, next_frame, original_frame_size);

        // Display the captured frame
        cv::imshow(windowName, next_frame);

        // Wait for 30ms or until 'q' key is pressed
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            break;
        }

        std::size_t frame_end = cv::getTickCount();

        double frame_fps = cv::getTickFrequency() / (frame_end - frame_start);
        double chpl_fps = cv::getTickFrequency() / chpl_delta;
        std::cout << "\rcv::FPS : \t " << frame_fps << " chpl::FPS : \t " << chpl_fps << "\t\r" << std::flush;
        last_frame_fps = frame_fps;
        last_chpl_fps = chpl_fps;
    }

    std::cout << "\nLast FPS: " << last_frame_fps << std::endl;
    std::cout << "Last CHPL FPS: " << last_chpl_fps << std::endl;
    std::cout << "Exiting webcam feed..." << std::endl;

    // Release the camera and destroy all windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


int main(int argc, char* argv[]) {
    chpl_library_init(argc, argv);
    
    chpl__init_Bridge(0, 0);
    chpl__init_smol(0, 0);

    globalLoadModel();

    int code = mirror();

    // std::size_t start = cv::getTickCount();
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    // std::size_t end = cv::getTickCount();
    // std::cout << "Total time taken: " << (end - start) / cv::getTickFrequency() << " seconds" << std::endl;


    chpl_library_finalize();
    return code;
}


