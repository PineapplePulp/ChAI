#include "smol_wrapper.h"


#include <opencv2/opencv.hpp>
#include <iostream>



cv::Mat new_frame(cv::Mat &frame) {
    // cv::Mat rgb_uchar_frame;
    // cv::cvtColor(frame, rgb_uchar_frame, cv::COLOR_BGR2RGB);

    // cv::Mat rgb_float_frame;
    // rgb_uchar_frame.convertTo(rgb_float_frame, CV_32FC3, 1.0f/255.0f);


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

    chpl_external_array 
        rgb_float_frame_data_ptr = chpl_make_external_array_ptr(rgb_float_frame.data,size);
    
    chpl_external_array 
        rgb_float_output_frame_array = getNewFrame(&rgb_float_frame_data_ptr, height, width, channels);
    
    chpl_free_external_array(rgb_float_frame_data_ptr);

    // cv::Mat new_rgb_frame(height, width, CV_8UC3,new_frame_array.elts);
    // cv::cvtColor(new_rgb_frame, new_rgb_frame, cv::COLOR_RGB2BGR);

    cv::Mat output_frame(height,width,CV_32FC3,rgb_float_output_frame_array.elts); // frame to write to
    output_frame.convertTo(output_frame, CV_8UC3, 255.0f); 
    cv::cvtColor(output_frame, output_frame, cv::COLOR_RGB2BGR);

    chpl_free_external_array(rgb_float_output_frame_array);

    return output_frame;

    // cv::Mat rgb_float_output_frame(height,width,CV_32FC3,rgb_float_output_frame_array.elts); // frame to write to

    // cv::Mat rgb_uchar_output_frame;
    // rgb_float_output_frame.convertTo(rgb_uchar_output_frame, CV_8UC3, 255.0f); 
    
    // cv::Mat bgr_uchar_output_frame;
    // cv::cvtColor(rgb_uchar_output_frame, bgr_uchar_output_frame, cv::COLOR_RGB2BGR);

    // return bgr_uchar_output_frame;
}

/*
cv::Mat new_frame(cv::Mat &frame) {
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    std::cout << "Frame size: " << rgb_frame.size() << std::endl;

    // cv::MatSize size = rgb_frame.size;
    // std::cout << "x " << size[0] << " y " << size[1] << " channels " << rgb_frame.dims << std::endl;
    int64_t width = rgb_frame.cols;
    int64_t height = rgb_frame.rows;
    int64_t channels = rgb_frame.channels();
    int64_t pixels = rgb_frame.total();
    int64_t size = pixels * channels;

    chpl_external_array frame_data_ptr = chpl_make_external_array_ptr(rgb_frame.data, size);
    chpl_external_array new_frame_array = getNewFrame(&frame_data_ptr, width, height, channels);
    chpl_free_external_array(frame_data_ptr);

    cv::Mat new_rgb_frame(height, width, CV_8UC3,new_frame_array.elts);
    cv::cvtColor(new_rgb_frame, new_rgb_frame, cv::COLOR_RGB2BGR);

    return new_rgb_frame;
}
*/

/*
void new_frame(cv::Mat &frame) {
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    std::cout << "Frame size: " << rgb_frame.size() << std::endl;

    // cv::MatSize size = rgb_frame.size;
    // std::cout << "x " << size[0] << " y " << size[1] << " channels " << rgb_frame.dims << std::endl;
    int64_t width = rgb_frame.cols;
    int64_t height = rgb_frame.rows;
    int64_t channels = rgb_frame.channels();
    int64_t size = rgb_frame.total() * channels;

    std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << ", Size: " << size << std::endl;

    // chpl_external_array frame_data_ptr = chpl_make_external_array_ptr(rgb_frame.data, );
}*/


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

    while (true) {

        uint64_t start = cv::getTickCount();

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

        cv::Mat next_frame = new_frame(frame);

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
    chpl_library_init(argc, argv);
    
    chpl__init_Bridge(0, 0);
    chpl__init_smol(0, 0);

    square(3);

    // int64_t array[4] = {1,2,3,4};
    // chpl_external_array array_ptr = chpl_make_external_array_ptr(&array,4);
    // int64_t sum = sumArray(&array_ptr);
    // chpl_free_external_array(array_ptr);
    // printf("sum: %d\n", sum);


    int64_t matrix[2][3] = { {1, 4, 2}, {3, 6, 8} };
    chpl_external_array matrix_ptr = chpl_make_external_array_ptr(matrix, 3 * 2);
    printArray(&matrix_ptr);
    chpl_free_external_array(matrix_ptr);

    globalLoadModel();

    int code = mirror();


    chpl_library_finalize();
    return code;
}


