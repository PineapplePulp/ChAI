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
# set(LIBTORCH_STATIC_INSTALL_DIR "${PROJECT_ROOT_DIR}/libtorch_static")
set(BRIDGE_DIR "${PROJECT_ROOT_DIR}/bridge")


find_package(chpl REQUIRED HINTS ${PROJECT_ROOT_DIR}/cmake/chapel)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake/chapel")

project(MyProject LANGUAGES CXX C CHPL)
message(STATUS "Using chpl: ${CMAKE_CHPL_COMPILER}")


if(APPLE)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
endif()
set(CMAKE_CXX_STANDARD 23)


include(LibTorchDL)
download_libtorch(
    CACHE_DIR ${PROJECT_CACHE_DIR}
    DESTINATION ${LIBTORCH_DIR}
)


include(FetchContent)
include(ExternalProject)


# if(NOT EXISTS "${LIBTORCH_STATIC_INSTALL_DIR}/lib/libtorch.a")
#     ExternalProject_Add(
#     pytorch
#     GIT_REPOSITORY      https://github.com/pytorch/pytorch.git
#     GIT_TAG             v2.6.0                # Example: specify a particular release
#     UPDATE_COMMAND      ""                     # Don’t auto-run 'git pull'
#     PATCH_COMMAND       ""                     # No custom patch step
    
#     # DOWNLOAD_DIR         "${CMAKE_BINARY_DIR}/pytorch-download" # Where to download the repo

#     # We need all PyTorch submodules. By default, ExternalProject won't do submodule init.
#     # So we can do that in a separate step if we want a full build. For a minimal CPU build,
#     # you might not need them all, but let's be safe:
#     STEP_TARGETS        clone
#     # After 'clone', run "git submodule update --init --recursive"
#     # to fetch all submodules.
#     # We can use a little trick with COMMAND.
#     #   PATCH_COMMAND       "git submodule update --init --recursive"

#     # CMAKE_ARGS
#     #     -DBUILD_SHARED_LIBS=OFF              # Build static libraries
#     #     -DBUILD_PYTHON=OFF                   # Don’t build Python bindings
#     #     -DBUILD_TEST=OFF                     # Don’t build tests
#     #     -DUSE_CUDA=OFF                       # Disable CUDA
#     #     -DUSE_CUDNN=OFF                      # Disable cuDNN
#     #     -DUSE_MKLDNN=OFF                     # Disable MKLDNN for simplicity
#     #     # -DBUILD_BINARY=ON
#     #     # -DUSE_DISTRIBUTED=ON
#     #     # -DBUILD_STATIC_RUNTIME_BENCHMARK=ON
#     #     # -DBUILD_LITE_INTERPRETER=ON
#     #     # -DUSE_STATIC_MKL=ON
#     #     # -DSTATIC_DISPATCH_BACKEND=ON
#     #     # -DCAFFE2_USE_MSVC_STATIC_RUNTIME=ON
#     #     # -DUSE_DISTRIBUTED=ON
#     #     # -DCMAKE_BUILD_TYPE=Release
#     #     -DCMAKE_INSTALL_PREFIX=${PYTORCH_INSTALL_DIR}
#     #     # -DCMAKE_POLICY_VERSION_MINIMUM=3.5
#         CMAKE_ARGS
#             -DBUILD_SHARED_LIBS=OFF              # Build static libraries
#             -DBUILD_PYTHON=OFF                   # Don’t build Python bindings
#             -DBUILD_TEST=OFF                     # Don’t build tests
#             -DUSE_CUDA=OFF                       # Disable CUDA
#             -DUSE_CUDNN=OFF                      # Disable cuDNN
#             -DUSE_MKLDNN=OFF                     # Disable MKLDNN for simplicity
#             -DCMAKE_BUILD_TYPE=Release
#             # -DSTATIC_DISPATCH_BACKEND=ON
#             -DCMAKE_INSTALL_PREFIX=${LIBTORCH_STATIC_INSTALL_DIR}
#             -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        
#     INSTALL_DIR         ${LIBTORCH_STATIC_INSTALL_DIR} # Where to install
#     )
# endif()


file(GLOB CHAI_LIB_FILES "${PROJECT_ROOT_DIR}/lib/*.chpl")

# file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/resources_dummy.c "int main(int argc, const char **argv){ return 1; }\n")
# add_executable(ChAI ${CMAKE_CURRENT_BINARY_DIR}/resources_dummy.c)
# # file(GLOB RESOURCE_FILES *.bmp *.wav moose.dat utf8.txt)
# foreach(RESOURCE_FILE ${CHAI_LIB_FILES})
#     add_custom_command(
#         TARGET ChAI 
#         POST_BUILD 
#         COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${RESOURCE_FILE} $<TARGET_FILE_DIR:ChAI>
#     )
# endforeach(RESOURCE_FILE)

# file(COPY ${CHAI_LIB_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})




add_custom_target(
    ChAI
    ALL
    DEPENDS ${PROJECT_SOURCE_DIR}/lib
    # SOURCES ${CHAI_LIB_FILES}
    # COMMAND ${CMAKE_COMMAND} -E echo "Building ChAI"
)

foreach(RESOURCE_FILE ${CHAI_LIB_FILES})
    add_custom_command(
        TARGET ChAI
        POST_BUILD
        # OUTPUT $<TARGET_FILE_DIR:ChAI>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RESOURCE_FILE} ${PROJECT_BINARY_DIR}/lib
        # DEPENDS ${CHAI_LIB_FILES}
    )
endforeach(RESOURCE_FILE)



# foreach(RESOURCE_FILE ${CHAI_LIB_FILES})
#     add_custom_command(
#         TARGET ChAI 
#         POST_BUILD 
#         COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${RESOURCE_FILE} $<TARGET_FILE_DIR:ChAI>
#     )
# endforeach(RESOURCE_FILE)


function(watch)
    set_property(
            DIRECTORY
            APPEND
            PROPERTY CMAKE_CONFIGURE_DEPENDS ${ARGV}
    )
endfunction()

# cmake_print_variables(CHAI_LIB_FILES)

add_library(bridge STATIC ${BRIDGE_DIR}/include/bridge.h ${BRIDGE_DIR}/lib/bridge.cpp)

target_include_directories(
    bridge
    PRIVATE
    ${BRIDGE_DIR}/include
    ${LIBTORCH_DIR}/include
    ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

set(BRIDGE_OBJECT_FILES $<TARGET_OBJECTS:bridge>)




file(GLOB LIBTORCH_ALL_LIB_FILES 
    "${LIBTORCH_DIR}/lib/*.a" 
    "${LIBTORCH_DIR}/lib/*.dylib" 
    "${LIBTORCH_DIR}/lib/*.so")

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
# cmake_print_variables(${BRIDGE_OBJECT_FILES})
# cmake_print_variables(BRIDGE_OBJECT_FILES)


add_executable(TorchBridge ${BRIDGE_DIR}/lib/Bridge.chpl)
add_dependencies(TorchBridge bridge)
add_dependencies(TorchBridge ChAI)
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




add_executable(TinyLayerTest 
    ${PROJECT_ROOT_DIR}/test/tiny/layer_test.chpl
    ${CHAI_LIB_FILES}
    )
add_dependencies(TinyLayerTest bridge)
add_dependencies(TinyLayerTest ChAI)
target_link_options(TinyLayerTest
    PRIVATE
        --main-module layer_test.chpl
        -M ${PROJECT_ROOT_DIR}/lib
        ${BRIDGE_DIR}/include/bridge.h
        ${BRIDGE_OBJECT_FILES}
        -L ${LIBTORCH_DIR}/lib
        ${LIBTORCH_LIBS_LINKER_ARGS}
        --ldflags "-Wl,-rpath,${LIBTORCH_DIR}/lib"
)
# chpl test/tiny/layer_test.chpl -M lib bridge/include/bridge.h build/CMakeFiles/bridge.dir/bridge/lib/bridge.cpp.o -L libtorch/lib -ltorch -ltorch_cpu -lc10 -ltorch_global_deps --ldflags "-Wl,-rpath,libtorch/lib"

# chpl --fast -o vgg test.chpl -M ../../lib /Users/iainmoncrief/Documents/Github/ChAI/bridge/include/bridge.h /Users/iainmoncrief/Documents/Github/ChAI/build/CMakeFiles/bridge.dir/bridge/lib/bridge.cpp.o -L /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib -ltorch -ltorch_cpu -lc10 -ltorch_global_deps --ldflags "-Wl,-rpath,/Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib"

# chpl -o vgg test.chpl $(../../embed_libtorch.sh .)
# chpl --fast -o vgg test.chpl $(../../embed_libtorch.sh .)




set(CHAI_LINKER_ARGS
    -M ${PROJECT_ROOT_DIR}/lib
    ${BRIDGE_DIR}/include/bridge.h
    ${BRIDGE_OBJECT_FILES}
    -L ${LIBTORCH_DIR}/lib
    ${LIBTORCH_LIBS_LINKER_ARGS}
    --ldflags "-Wl,-rpath,${LIBTORCH_DIR}/lib"
)



add_executable(vgg 
    "${PROJECT_ROOT_DIR}/examples/vgg/test.chpl"
    ${PROJECT_ROOT_DIR}/examples/vgg/VGG.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(vgg bridge)
add_dependencies(vgg ChAI)
target_link_options(vgg
    PRIVATE
    # -M ${PROJECT_ROOT_DIR}/examples/vgg
    -svggExampleDir="${PROJECT_ROOT_DIR}/examples/vgg"
    ${CHAI_LINKER_ARGS}
)

add_custom_command(
    TARGET vgg
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_ROOT_DIR}/examples/vgg/images"
            "$<TARGET_FILE_DIR:vgg>/images"
    COMMENT "Copying ${PROJECT_ROOT_DIR}/examples/vgg/images to $<TARGET_FILE_DIR:vgg>/images"
)

# ./vgg images/frog.jpg

add_subdirectory(examples)
add_subdirectory("test")
