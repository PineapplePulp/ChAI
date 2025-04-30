# ────────────────────────────────
# 1. Build stage
# ────────────────────────────────
FROM ubuntu:24.04 AS builder



ENV DEBIAN_FRONTEND=noninteractive

COPY deb /

# -------- toolchain & helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ cmake ninja-build \
        git curl wget ca-certificates unzip \
        python3 python3-pip libssl-dev \
        apt-utils \
    && ((apt-get install -y --no-install-recommends ./chapel-2.4.0-1.ubuntu24.arm64.deb) || (apt-get install -y --no-install-recommends ./chapel-2.4.0-1.ubuntu24.amd64.deb)) \
    && rm -rf /var/lib/apt/lists/*

#     && (apt-get install -y --no-install-recommends ./chapel-2.4.0-1.ubuntu24.arm64.deb) || (apt install -y ./chapel-2.4.0-1.ubuntu24.amd64.deb)

# -------- LibTorch (change CUDA tag for GPU builds)
# ARG TORCH_VERSION=2.2.0         # ← keep in sync with your sources
# ARG TORCH_VARIANT=cpu           # cpu | cu118 | cu121 …
# RUN mkdir -p /opt \
#  && cd /opt \
#  && wget -q --show-progress \
#       https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_VARIANT}.zip \
#  && unzip libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+${TORCH_VARIANT}.zip \
#  && rm libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+${TORCH_VARIANT}.zip


RUN mkdir -p /app \
    && cd /app \
    && wget -q --show-progress \
        https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-latest.zip \
    && mkdir -p /app/.cache \
    && mv /app/libtorch-cxx11-abi-shared-with-deps-latest.zip /app/.cache/libtorch_cache.zip


ENV Torch_DIR=/app/libtorch
ENV CMAKE_PREFIX_PATH=/app/libtorch/libtorch/share/cmake
# lets CMake find TorchConfig.cmake


# -------- build your project
WORKDIR /app
COPY . /app

# (uncomment if you have submodules)
# RUN git submodule update --init --recursive

RUN ls

RUN cd /app \
    && mkdir -p build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Debug .. \
    && make MyExample

# RUN mkdir build \
#     && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Debug .. \
#     && make TinyLayerTest -j$(nproc)

#RUN cd build && ./TinyLayerTest

    # && cmake -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release .. \

# configure-and-build
# RUN cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
#  && cmake --build build --target all -j$(nproc)

# RUN cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release

# RUN cmake --build build --target all -j$(nproc)

