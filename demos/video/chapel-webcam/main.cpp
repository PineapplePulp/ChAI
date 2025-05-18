#include "smol_wrapper.h"


#include <opencv2/opencv.hpp>
#include <iostream>



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

    while (true) {
        // Capture a new frame from webcam
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured.\n";
            break;
        }
        cv::Mat next_frame = new_frame(frame);
        // Display the captured frame
        cv::imshow(windowName, next_frame);

        // Wait for 30ms or until 'q' key is pressed
        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            break;
        }
    }

    // Release the camera and destroy all windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


int main(int argc, char* argv[]) {
    chpl_library_init(argc, argv);

    square(3);

    int64_t array[4] = {1,2,3,4};
    chpl_external_array array_ptr = chpl_make_external_array_ptr(&array,4);
    int64_t sum = sumArray(&array_ptr);
    chpl_free_external_array(array_ptr);
    printf("sum: %d\n", sum);


    int64_t matrix[2][3] = { {1, 4, 2}, {3, 6, 8} };
    chpl_external_array matrix_ptr = chpl_make_external_array_ptr(matrix, 3 * 2);
    printArray(&matrix_ptr);
    chpl_free_external_array(matrix_ptr);

    int code = mirror();


    chpl_library_finalize();
    return code;
}


