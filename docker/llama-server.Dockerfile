# llama-server image built by `llmctl build`. Embedded in the llmctl binary.
#
# The build context is the llama.cpp (or fork) source tree, fetched on the
# host by llmctl. Build args are set by llmctl from the `builds` config:
#
#   BUILD_IMAGE    CUDA devel image (desktop) or L4T JetPack image (Jetson)
#   RUNTIME_IMAGE  CUDA runtime image, or the same JetPack image on Jetson
#   CUDA_ARCH      CMAKE_CUDA_ARCHITECTURES, e.g. "89" or "86;89" or "87" (Orin)
#   CMAKE_ARGS     extra cmake flags (space separated)
#   TARGET         cmake target / binary name (default llama-server)
#   CCACHE_ID      BuildKit cache id so CUDA and Jetson caches stay separate
#
# Requires BuildKit (default since Docker 23) for the ccache cache mount.

ARG BUILD_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04

FROM ${BUILD_IMAGE} AS build
ARG CUDA_ARCH
ARG CMAKE_ARGS=""
ARG TARGET=llama-server
ARG CCACHE_ID=llmctl-ccache

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:${PATH} \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=20G

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build ccache git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Static llama-server: no shared ggml/llama libs to copy into the runtime
# image. libcuda comes from the driver at run time (--allow-shlib-undefined,
# as in upstream's .devops/cuda.Dockerfile).
RUN --mount=type=cache,id=${CCACHE_ID},target=/ccache \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      ${CMAKE_ARGS} \
 && cmake --build build --target "${TARGET}" -j"$(nproc)" \
 && mkdir -p /out \
 && cp "build/bin/${TARGET}" /out/llama-server \
 && (ccache -s || true)

FROM ${RUNTIME_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/llama-server /app/llama-server
ENTRYPOINT ["/app/llama-server"]
