#include "mirror.h"
// #include <SDL2/SDL.h>
#include <opencv2/opencv.hpp>
#include <iostream>



void displayMirror(cv::VideoCapture &cap, const std::string& windowName) {
    cv::Mat frame;
    while (true) {
        // Capture a new frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Show the frame in the window
        cv::imshow(windowName, frame);


        // Wait for 30ms. Exit if any key is pressed.
        if (cv::waitKey(30) >= 0) {
            std::cout << "Key pressed, exiting..." << std::endl;
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}


// int displayMirrorLoopSDL() {
//     if (SDL_Init(SDL_INIT_VIDEO) < 0) {
//         SDL_Log("Could not initialize SDL: %s", SDL_GetError());
//         return -1;
//     }

//     // 2) Open default webcam via OpenCV
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         SDL_Log("Could not open webcam");
//         SDL_Quit();
//         return -1;
//     }

//     // Get camera resolution
//     int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//     int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

//     // 3) Create a borderless SDL2 window
//     SDL_Window* window = SDL_CreateWindow(
//         "Webcam",
//         SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
//         w, h,
//         SDL_WINDOW_BORDERLESS | SDL_WINDOW_SHOWN
//     );

//     SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
//     // Create a streaming texture in RGB24 format
//     SDL_Texture* texture = SDL_CreateTexture(
//         renderer,
//         SDL_PIXELFORMAT_RGB24,
//         SDL_TEXTUREACCESS_STREAMING,
//         w, h
//     );

//     // 4) Main loop: grab frame, convert, update texture, render
//     bool running = true;
//     SDL_Event ev;
//     while (running) {
//         // Handle events
//         while (SDL_PollEvent(&ev)) {
//             if (ev.type == SDL_QUIT) {
//                 running = false;
//             }
//         }

//         // Grab frame (BGR), convert to RGB
//         cv::Mat frame;
//         cap >> frame;
//         if (frame.empty()) break;
//         cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

//         // Update SDL texture with the raw pixel data
//         SDL_UpdateTexture(texture, nullptr, frame.data, frame.step);

//         // Render it
//         SDL_RenderClear(renderer);
//         SDL_RenderCopy(renderer, texture, nullptr, nullptr);
//         SDL_RenderPresent(renderer);
//     }

//     // Cleanup
//     SDL_DestroyTexture(texture);
//     SDL_DestroyRenderer(renderer);
//     SDL_DestroyWindow(window);
//     SDL_Quit();
//     return 0;
// }

// extern "C" void run_mirror() asm ("run_mirror");
extern "C" int run_mirror(void) {

    // return displayMirrorLoopSDL();

    // cv::VideoCapture cap(0); // Open the default camera
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Could not open camera" << std::endl;
    //     return -1;
    // }

    // // Create a window to display the video
    // const std::string windowName = "Mirror";
    // cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // // // Start displaying the video
    // displayMirror(cap, windowName);

    // return 0;


        cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    // Create a window to display the video
    const std::string windowName = "Mirror";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        // Capture a new frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Show the frame in the window
        cv::imshow(windowName, frame);


        // Wait for 30ms. Exit if any key is pressed.
        if (cv::waitKey(30) >= 0) {
            std::cout << "Key pressed, exiting..." << std::endl;
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

