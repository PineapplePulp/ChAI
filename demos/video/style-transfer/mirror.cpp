#include "mirror.h"

#include <opencv2/opencv.hpp>
#include <iostream>

// extern "C" void run_mirror() asm ("run_mirror");
extern "C" int run_mirror(void) {
    // cv::VideoCapture cap(0); // Open the default camera (0)
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Could not open camera." << std::endl;
    // }

    // cv::Mat frame;
    // while (true) {
    //     cap >> frame; // Capture a new frame
    //     if (frame.empty()) {
    //         std::cerr << "Error: Could not capture frame." << std::endl;
    //         break;
    //     }

    //     cv::imshow("Webcam", frame); // Display the captured frame
    //     if (cv::waitKey(30) >= 0) break; // Exit on any key press
    // }

    // cap.release(); // Release the camera
    // cv::destroyAllWindows(); // Close all OpenCV windows

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    const std::string window_name = "Webcam";

    // Create a window to display the video
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    while (true) {
        // Capture a new frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Show the frame in the window
        cv::imshow(window_name, frame);

        // Wait for 30ms. Exit if any key is pressed.
        if (cv::waitKey(30) >= 0) break;
    }

    // Release the camera and destroy the window
    cap.release();
    cv::destroyAllWindows();

    return 0;

}

