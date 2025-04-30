

# export OpenCV_DIR="$(brew --prefix opencv)/share/opencv4"
# /opt/homebrew/opt/opencv/share/opencv4
find_package(OpenCV 4 REQUIRED)

find_library(ACCELERATE Accelerate REQUIRED)
find_library(METAL Metal REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)


add_executable(VidStreamer
    ${CMAKE_CURRENT_SOURCE_DIR}/webcam_infer.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/cvutil.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imageops.hpp
)

target_include_directories(VidStreamer
    PRIVATE
        ${LIBTORCH_DIR}/include
        ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

target_link_directories(VidStreamer
    PRIVATE
        ${LIBTORCH_DIR}/lib
)

target_link_libraries(VidStreamer
    PRIVATE 
        -ltorch
        -ltorch_cpu
        -lc10
        -ltorch_global_deps
        ${OpenCV_LIBS}
        # ${TORCH_LIBRARIES}
        ${ACCELERATE}
        ${METAL}
        ${FOUNDATION}
        BridgeUtil
)

set_target_properties(VidStreamer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(VidStreamer PRIVATE -Ofast -flto -ffast-math)
  target_link_options(VidStreamer PRIVATE -flto)
endif()


