

# export OpenCV_DIR="$(brew --prefix opencv)/share/opencv4"
# /opt/homebrew/opt/opencv/share/opencv4




# This is messy
find_package(OpenCV 4 REQUIRED)

find_library(ACCELERATE Accelerate REQUIRED)
find_library(METAL Metal REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)



add_executable(VidStreamer
    ${CMAKE_CURRENT_SOURCE_DIR}/webcam-capture/webcam_infer.cpp
)

target_include_directories(VidStreamer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LIBTORCH_DIR}/include
        ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

target_link_directories(VidStreamer PRIVATE ${LIBTORCH_DIR}/lib)

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
)

set_target_properties(VidStreamer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(VidStreamer PRIVATE -Ofast -flto -ffast-math)
  target_link_options(VidStreamer PRIVATE -flto)
endif()







add_executable(StyleTransfer
    ${CMAKE_CURRENT_SOURCE_DIR}/style-transfer/style_transfer.cpp
)

target_include_directories(StyleTransfer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LIBTORCH_DIR}/include
        ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

target_link_directories(StyleTransfer PRIVATE ${LIBTORCH_DIR}/lib)

target_link_libraries(StyleTransfer
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
)

set_target_properties(StyleTransfer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(StyleTransfer PRIVATE -Ofast -flto -ffast-math)
  target_link_options(StyleTransfer PRIVATE -flto)
endif()


add_custom_command(
    TARGET StyleTransfer
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/style-transfer/models"
            "$<TARGET_FILE_DIR:StyleTransfer>/style-transfer/models"
    COMMENT "NOT! Copying ${PROJECT_ROOT_DIR}/examples/vgg/images to $<TARGET_FILE_DIR:vgg>/images"
)