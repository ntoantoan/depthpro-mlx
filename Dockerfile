# syntax=docker/dockerfile:1.7
#
# depthpro-mlx on Linux (MLX CPU or CUDA backend).
#
#   docker build -t depthpro-mlx .                               # CPU, amd64 or arm64
#   docker build -t depthpro-mlx:cuda --build-arg MLX_BACKEND=cuda12 \
#       --build-arg BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 .
#   docker run --rm -v "$PWD:/data" -v depthpro-cache:/cache depthpro-mlx photo.jpg --web
#
# On a Mac, Docker runs a Linux VM without Metal: use the native package there
# (pip install depthpro-mlx), which is much faster. This image is for Linux servers,
# CI and reproducible batch jobs.

ARG BASE=python:3.12-slim-bookworm
ARG MLX_BACKEND=cpu

# ---------------------------------------------------------------- build stage
FROM ghcr.io/astral-sh/uv:0.7.12 AS uv
FROM ${BASE} AS build
ARG MLX_BACKEND
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# CUDA base images ship without Python: let uv provide one there.
RUN if ! command -v python3 >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv && rm -rf /var/lib/apt/lists/*; \
    fi
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python "$(command -v python3.12 || command -v python3)" && \
    VIRTUAL_ENV=/opt/venv uv pip install ".[inpaint]" "mlx[${MLX_BACKEND}]"

# -------------------------------------------------------------- runtime stage
FROM ${BASE} AS runtime
ARG MLX_BACKEND
LABEL org.opencontainers.image.title="depthpro-mlx" \
      org.opencontainers.image.description="Apple Depth Pro on MLX: metric depth, 3D, refocus, video depth" \
      org.opencontainers.image.source="https://github.com/ntoantoan/depthpro-mlx" \
      org.opencontainers.image.licenses="MIT" \
      io.depthpro-mlx.backend="${MLX_BACKEND}"
RUN if ! command -v python3 >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends python3.12 && rm -rf /var/lib/apt/lists/*; \
    fi && \
    useradd --create-home --uid 1000 depthpro && mkdir -p /cache /data && chown depthpro /cache /data
COPY --from=build /opt/venv /opt/venv
# mlx-cpu bundles the netlib *reference* BLAS: single-threaded, ~7 GFLOP/s. Swapping in OpenBLAS
# (same BLAS/LAPACK ABI) makes float32 matmul ~40x faster, which is what Depth Pro spends its time on.
RUN if [ "$MLX_BACKEND" = "cpu" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libopenblas0-pthread && rm -rf /var/lib/apt/lists/* && \
      libs=$(echo /opt/venv/lib/python3.*/site-packages/mlx_cpu.libs) && \
      cp /usr/lib/*-linux-gnu/openblas-pthread/libblas.so.3 "$libs"/libblas-*.so.3.10.0 && \
      cp /usr/lib/*-linux-gnu/openblas-pthread/liblapack.so.3 "$libs"/liblapack-*.so.3.10.0; \
    fi
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1
USER depthpro
WORKDIR /data
VOLUME ["/cache"]
HEALTHCHECK NONE
ENTRYPOINT ["depthpro-mlx"]
CMD ["--help"]
