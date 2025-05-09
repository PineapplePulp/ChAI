#include <opencv2/opencv.hpp>
#include <iostream>
#include "mirror.h"

extern "C" void run_mirror() {
    cv::VideoCapture cap(0); // Open the default camera (0)
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // Capture a new frame
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        cv::imshow("Webcam", frame); // Display the captured frame
        if (cv::waitKey(30) >= 0) break; // Exit on any key press
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows
}

