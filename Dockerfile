# syntax=docker/dockerfile:1.7
# CUDA 12.9 runtime: works with the server driver (CUDA 12.9) and with the RTX 5080
# (Blackwell needs CUDA >= 12.8). Uses cupy-cuda12x (not 13x): the CuPy wheel must
# match the CUDA major version of this image.
# CUDA 13 images would require a host driver >= 580.
# FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.14.4
FROM python:${PYTHON_VERSION}-slim


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Only the system libraries needed at runtime (OpenMP for xgboost, CA certs for uv).
# Python itself is installed by uv, so the version matches the one used in development.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: fast installer that also downloads the exact Python version.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Python goes in /opt/python (readable by any UID, unlike /root), cache on the same
# filesystem rules as the BuildKit cache mount (copy instead of hardlink).
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy

# Isolated Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}
# RUN uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}-slim
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# CuPy compiles kernels at runtime (NVRTC) and caches them: keep the cache in a path
# writable by any UID (useful when running with `--user $(id -u):$(id -g)`).
ENV CUPY_CACHE_DIR=/tmp/cupy_cache

WORKDIR /app

# Dependencies first, for better layer caching.
# requirements.txt is frozen from the development venv and pins cupy-cuda13x, which
# does not match this CUDA 12.9 image. Here it is swapped for cupy-cuda12x[ctk] at the
# same version: [ctk] installs the CUDA headers/NVRTC as pip packages (the runtime
# base image has none), so CuPy can compile kernels such as cp.asnumpy on
# non-contiguous arrays. requirements.txt may be UTF-16 (PowerShell `pip freeze >`),
# so it is converted to UTF-8 first. The greps fail the build if the swap did not happen,
# so two CuPy packages can never end up in the same venv.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$(head -c2 requirements.txt | od -An -tx1 | tr -d ' \n')" = "fffe" ]; then \
        iconv -f UTF-16 -t UTF-8 requirements.txt > /tmp/req.utf8; \
    else \
        cp requirements.txt /tmp/req.utf8; \
    fi \
    && tr -d '\r' < /tmp/req.utf8 \
        | sed -E 's/^cupy-cuda13x==/cupy-cuda12x[ctk]==/' \
        | sed -E 's/^xgboost==/xgboost-cu12==/' \
        > /tmp/requirements.docker.txt \
    && grep -q '^cupy-cuda12x\[ctk\]==' /tmp/requirements.docker.txt \
    && grep -q '^xgboost-cu12==' /tmp/requirements.docker.txt \
    && ! grep -q 'cuda13x' /tmp/requirements.docker.txt \
    && uv pip install -r /tmp/requirements.docker.txt
#    && tr -d '\r' < /tmp/req.utf8 \
#        | sed -E 's/^cupy-cuda13x==/cupy-cuda12x[ctk]==/' \
#        > /tmp/requirements.docker.txt \
#    && grep -q '^cupy-cuda12x\[ctk\]==' /tmp/requirements.docker.txt \
#    && ! grep -q 'cuda13x' /tmp/requirements.docker.txt \
#    && uv pip install -r /tmp/requirements.docker.txt

# Application code (see .dockerignore: no data, results, venv, git, IDE files)
COPY . .

# Mount points used at runtime
RUN mkdir -p /app/config /app/data /app/results

# Arguments are supplied at `docker run`, e.g.: docker run ... image --runs 100
ENTRYPOINT ["python", "main.py"]