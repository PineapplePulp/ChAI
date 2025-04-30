cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MyProject)
set(CMAKE_CXX_STANDARD 17)
# list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake")
find_package(Torch REQUIRED)  
# find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")
add_executable(MyProject torch_test.cpp)

target_link_libraries(MyProject ${TORCH_LIBRARIES})
set_property(TARGET MyProject PROPERTY CXX_STANDARD 17)

