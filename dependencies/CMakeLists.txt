cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX)

# For ExternalProject_Add
include(FetchContent)
include(ExternalProject)

# ------------------------------------------------------------------------------
# 1) External project: build PyTorch (libtorch) from source as STATIC
# ------------------------------------------------------------------------------

# Where to place PyTorch after installation
set(PYTORCH_INSTALL_DIR "${CMAKE_BINARY_DIR}/pytorch-install")

# ExternalProject_Add can fetch from Git, a local path, or a release tarball.
# Here, for simplicity, we'll fetch from Git. In practice, you might want
# a fixed commit or a release tarball for reproducible builds.

ExternalProject_Add(
  pytorch
  GIT_REPOSITORY      https://github.com/pytorch/pytorch.git
  GIT_TAG             v2.0.1                # Example: specify a particular release
  UPDATE_COMMAND      ""                     # Don’t auto-run 'git pull'
  PATCH_COMMAND       ""                     # No custom patch step

  # We need all PyTorch submodules. By default, ExternalProject won't do submodule init.
  # So we can do that in a separate step if we want a full build. For a minimal CPU build,
  # you might not need them all, but let's be safe:
  STEP_TARGETS        clone
  # After 'clone', run "git submodule update --init --recursive"
  # to fetch all submodules.
  # We can use a little trick with COMMAND.
  PATCH_COMMAND       "git submodule update --init --recursive"

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
)

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

    # System libraries often needed:
    pthread
    dl
    rt
)

# ------------------------------------------------------------------------------
# 3) Build your own executable that uses torch_interface
# ------------------------------------------------------------------------------

# add_executable(my_app src/main.cpp)

# # Link your app against our interface library
# target_link_libraries(my_app PRIVATE torch_interface)

