cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# project(MyProject LANGUAGES CXX)

include(CMakePrintHelpers)

# set(CMAKE_VERBOSE_MAKEFILE ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 17)


set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_BINARY_DIR "${CMAKE_BINARY_DIR}")

find_package(chpl REQUIRED HINTS ${PROJECT_ROOT_DIR}/cmake/chapel)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake/chapel")

project(MyProject LANGUAGES CXX C CHPL)


# For ExternalProject_Add
include(FetchContent)
include(ExternalProject)

# ------------------------------------------------------------------------------
# 1) External project: build PyTorch (libtorch) from source as STATIC
# ------------------------------------------------------------------------------

# Where to place PyTorch after installation
set(PYTORCH_INSTALL_DIR "${PROJECT_BINARY_DIR}/pytorch-install")

# ExternalProject_Add can fetch from Git, a local path, or a release tarball.
# Here, for simplicity, we'll fetch from Git. In practice, you might want
# a fixed commit or a release tarball for reproducible builds.

ExternalProject_Add(
  pytorch
  GIT_REPOSITORY      https://github.com/pytorch/pytorch.git
  GIT_TAG             v2.6.0                # Example: specify a particular release
  UPDATE_COMMAND      ""                     # Don’t auto-run 'git pull'
  PATCH_COMMAND       ""                     # No custom patch step
  
  # DOWNLOAD_DIR         "${CMAKE_BINARY_DIR}/pytorch-download" # Where to download the repo

  # We need all PyTorch submodules. By default, ExternalProject won't do submodule init.
  # So we can do that in a separate step if we want a full build. For a minimal CPU build,
  # you might not need them all, but let's be safe:
  STEP_TARGETS        clone
  # After 'clone', run "git submodule update --init --recursive"
  # to fetch all submodules.
  # We can use a little trick with COMMAND.
#   PATCH_COMMAND       "git submodule update --init --recursive"

  CMAKE_ARGS
      -DBUILD_SHARED_LIBS=OFF              # Build static libraries
      -DBUILD_PYTHON=OFF                   # Don’t build Python bindings
      -DBUILD_TEST=OFF                     # Don’t build tests
      -DUSE_CUDA=OFF                       # Disable CUDA
      -DUSE_CUDNN=OFF                      # Disable cuDNN
      -DUSE_MKLDNN=OFF                     # Disable MKLDNN for simplicity
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${PYTORCH_INSTALL_DIR}

  INSTALL_DIR         ${PYTORCH_INSTALL_DIR} # Where to install

  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_PATCH ON
  LOG_CONFIGURE ON
  # LOG_BUILD ON
  LOG_INSTALL ON
  
)


file(GLOB_RECURSE PYTORCH_INCLUDES "${PYTORCH_INSTALL_DIR}/include" "*.h")
file(GLOB_RECURSE PYTORCH_LIBS "${PYTORCH_INSTALL_DIR}/lib" "*.a")


# ------------------------------------------------------------------------------
# 2) Create an INTERFACE library to wrap the installed static libs
# ------------------------------------------------------------------------------

# We'll create a dummy target that depends on 'pytorch' so that
# building your own code will first build/install PyTorch.

add_library(torch_interface INTERFACE)

# Ensure that our 'torch_interface' target isn't used until PyTorch is built
add_dependencies(torch_interface pytorch)

# Include directories for PyTorch
target_include_directories(torch_interface INTERFACE
    "${PYTORCH_INSTALL_DIR}/include"
    "${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
)

# Link the relevant static libraries.  For a minimal CPU-only build, 
# you'll likely need at least these (names can vary by version).
# The exact set can differ depending on which components got built.

target_link_libraries(torch_interface INTERFACE
    "${PYTORCH_INSTALL_DIR}/lib/libtorch.a"
    "${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.a"
    "${PYTORCH_INSTALL_DIR}/lib/libc10.a"

    ${PYTORCH_LIBS}

    # System libraries often needed:
    pthread
    dl
    rt
)



add_executable(CHPLTest lib/CHPLTest.chpl)


add_library(torchbridge STATIC "${PROJECT_ROOT_DIR}/lib/bridge.cpp" "${PROJECT_ROOT_DIR}/include/bridge.h")
add_dependencies(torchbridge torch_interface)

target_include_directories(torchbridge PUBLIC
    "${PYTORCH_INSTALL_DIR}/include"
    "${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
    "${PROJECT_ROOT_DIR}/include"
)
# target_link_directories(torchbridge PUBLIC
#     "${PYTORCH_INSTALL_DIR}/lib"
# )
target_link_libraries(torchbridge 
    torch_interface
    ${PYTORCH_LIBS}
    )





add_executable(Bridge lib/Bridge.chpl include/bridge.h)
# add_dependencies(TorchBridge torchbridge)

target_link_options(Bridge 
  PRIVATE 
    "${PROJECT_ROOT_DIR}/include/bridge.h"
    "-L${PROJECT_BINARY_DIR}"
    -L. 
    "-ltorchbridge"
    "-I${PROJECT_BINARY_DIR}"
    "-L${PYTORCH_INSTALL_DIR}/lib"
    "-I${PYTORCH_INSTALL_DIR}/include"
    "-I${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
    "-ltorch"
    "-ltorch_cpu"
    "-lcpuinfo"
    "-lc10"
    "-lpthread"
    "-ldl"
  )



# ------------------------------------------------------------------------------
# 3) Build your own executable that uses torch_interface
# ------------------------------------------------------------------------------

# add_executable(my_app src/main.cpp)

# # Link your app against our interface library
# target_link_libraries(my_app PRIVATE torch_interface)

