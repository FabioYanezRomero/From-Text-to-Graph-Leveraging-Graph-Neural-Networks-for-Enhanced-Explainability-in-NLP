# Use NVIDIA CUDA 11.6 with Ubuntu 20.04 as base image to ensure glibc 2.28+ and GPU support
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    procps \
    lsof \
    net-tools \
    iproute2 \
    build-essential \
    ca-certificates \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy application code
COPY . .

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]
