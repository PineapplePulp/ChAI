cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# project(MyProject LANGUAGES CXX)

include(CMakePrintHelpers)

# set(CMAKE_VERBOSE_MAKEFILE ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(PROJECT_CACHE_DIR "${PROJECT_ROOT_DIR}/.cache")
set(LIBTORCH_DIR "${PROJECT_ROOT_DIR}/libtorch")
set(BRIDGE_DIR "${PROJECT_ROOT_DIR}/bridge")


find_package(chpl REQUIRED HINTS ${PROJECT_ROOT_DIR}/cmake/chapel)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake/chapel")

project(MyProject LANGUAGES CXX C CHPL)
message(STATUS "Using chpl: ${CMAKE_CHPL_COMPILER}")

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_STANDARD 17)


include(LibTorchDL)
download_libtorch(
    CACHE_DIR ${PROJECT_CACHE_DIR}
    DESTINATION ${LIBTORCH_DIR}
)


add_library(bridge OBJECT ${BRIDGE_DIR}/include/bridge.h ${BRIDGE_DIR}/lib/bridge.cpp)

target_include_directories(
    bridge
    PUBLIC
    ${BRIDGE_DIR}/include
    ${LIBTORCH_DIR}/include
    ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

set(BRIDGE_OBJECT_FILES $<TARGET_OBJECTS:bridge>)


add_executable(TorchBridge ${BRIDGE_DIR}/lib/Bridge.chpl)
add_dependencies(TorchBridge bridge)
target_link_options(TorchBridge
    PRIVATE
        ${BRIDGE_DIR}/include/bridge.h
        ${BRIDGE_OBJECT_FILES}
        -L ${LIBTORCH_DIR}/lib
        "-ltorch"
        "-ltorch_cpu"
        "-lc10"
        --ldflags "-Wl,-rpath,${LIBTORCH_DIR}/lib"
)
install(TARGETS TorchBridge DESTINATION ".")
