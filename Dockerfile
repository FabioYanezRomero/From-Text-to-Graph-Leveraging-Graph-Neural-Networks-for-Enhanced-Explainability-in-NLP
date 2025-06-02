# Use NVIDIA CUDA 11.6 with Ubuntu 20.04 as base image to ensure glibc 2.28+ and GPU support
FROM ubuntu:20.04

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

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio && \
    # Install other dependencies from requirements.txt (excluding PyTorch)
    grep -v -E '^torch|^torchvision|^torchaudio' requirements.txt > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && \
    # Install PyTorch Geometric dependencies with CUDA support
    pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html && \
    pip install torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html && \
    pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html && \
    pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html && \
    pip install torch-geometric==2.3.0

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
