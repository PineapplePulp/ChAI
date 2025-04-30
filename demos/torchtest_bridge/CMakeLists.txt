# cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
include(CMakePrintHelpers)

# project(MyProject)
# set(CMAKE_CXX_STANDARD 17)
# list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake")
find_package(Torch REQUIRED)  
# find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -lm -ldl")

cmake_print_variables(TORCH_LIBRARIES)
cmake_print_variables(TORCH_INCLUDE_DIRS)
cmake_print_variables(TORCH_INSTALL_PREFIX)
cmake_print_variables(TORCH_CXX_FLAGS)
cmake_print_variables(TORCH_LIBRARY)

add_executable(TorchTestBridge ${CMAKE_CURRENT_SOURCE_DIR}/torch_test.cpp)

target_link_libraries(TorchTestBridge ${TORCH_LIBRARIES})
set_property(TARGET TorchTestBridge PROPERTY CXX_STANDARD 23)

set_property(TARGET TorchTestBridge PROPERTY
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)