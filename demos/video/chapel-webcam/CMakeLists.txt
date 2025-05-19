# This is messy
find_package(OpenCV 4 REQUIRED)

find_library(ACCELERATE Accelerate REQUIRED)
find_library(METAL Metal REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/lib/smol.cmake)

add_executable(ChapelWebcam
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(ChapelWebcam
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LIBTORCH_DIR}/include
        ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

target_link_directories(ChapelWebcam PRIVATE ${LIBTORCH_DIR}/lib)

target_link_libraries(ChapelWebcam
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

target_include_directories(ChapelWebcam PUBLIC ${smol_INCLUDE_DIRS})
target_link_libraries(ChapelWebcam PUBLIC ${smol_LINK_LIBS})


set_target_properties(ChapelWebcam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(ChapelWebcam PRIVATE -Ofast -flto -ffast-math)
  target_link_options(ChapelWebcam PRIVATE -flto)
endif()
