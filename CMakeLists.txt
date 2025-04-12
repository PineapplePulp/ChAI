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




file(GLOB LIBTORCH_ALL_LIB_FILES "${LIBTORCH_DIR}/lib/*.a" "${LIBTORCH_DIR}/lib/*.dylib" "${LIBTORCH_DIR}/lib/*.so")

set(LIBTORCH_ALL_LIBS "")
foreach(lib_path IN LISTS LIBTORCH_ALL_LIB_FILES)
    get_filename_component(lib_name "${lib_path}" NAME_WE)
    list(APPEND LIBTORCH_ALL_LIBS "${lib_name}")
endforeach()


set(REQUIRED_LIBS
    "libtorch"
    "libtorch_cpu"
    "libc10"
    "libtorch_global_deps"
)

set(DISALLOWED_LIBS
    "libtorch_python"
)

set(LIBTORCH_LIBS_LINKER_ARGS "")  # Will hold the list of "-l..." flags.
foreach(lib_name IN LISTS LIBTORCH_ALL_LIBS)
    if(lib_name IN_LIST DISALLOWED_LIBS)
        if(lib_name IN_LIST REQUIRED_LIBS)
            message(FATAL_ERROR "Required lib ${lib_name} is disallowed.")
        else()
            message(STATUS "Skipping disallowed lib: ${lib_name}")
            continue()
        endif()
    endif()
    string(REGEX REPLACE "^lib" "" lib_name_short "${lib_name}")
    list(APPEND LIBTORCH_LIBS_LINKER_ARGS "-l${lib_name_short}")
endforeach()

# cmake_print_variables(LIBTORCH_LIBS_LINKER_ARGS)

add_executable(TorchBridge ${BRIDGE_DIR}/lib/Bridge.chpl)
add_dependencies(TorchBridge bridge)
target_link_options(TorchBridge
    PRIVATE
        ${BRIDGE_DIR}/include/bridge.h
        ${BRIDGE_OBJECT_FILES}
        -L ${LIBTORCH_DIR}/lib
        # "-ltorch"
        # "-ltorch_cpu"
        # "-lc10"
        # "-ltorch_global_deps"
        ${LIBTORCH_LIBS_LINKER_ARGS}
        --ldflags "-Wl,-rpath,${LIBTORCH_DIR}/lib"
)
# install(TARGETS TorchBridge DESTINATION ".")
