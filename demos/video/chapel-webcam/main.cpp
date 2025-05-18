#include "smol_wrapper.h"


// #include <opencv2/opencv.hpp>
// #include <iostream>


// int mirror() {
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         std::cerr << "Error: Cannot open the webcam.\n";
//         return -1;
//     }

//     cv::Mat frame;
//     const std::string windowName = "Webcam Feed";
//     cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

//     while (true) {
//         // Capture a new frame from webcam
//         cap >> frame;
//         if (frame.empty()) {
//             std::cerr << "Error: Empty frame captured.\n";
//             break;
//         }

//         // Display the captured frame
//         cv::imshow(windowName, frame);

//         // Wait for 30ms or until 'q' key is pressed
//         char key = static_cast<char>(cv::waitKey(30));
//         if (key == 'q' || key == 27) { // 'q' or ESC to quit
//             break;
//         }
//     }

//     // Release the camera and destroy all windows
//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }

int mirror() { return 0; }

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


