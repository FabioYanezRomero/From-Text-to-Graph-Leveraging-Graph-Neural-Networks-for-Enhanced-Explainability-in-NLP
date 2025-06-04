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

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch first
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and its dependencies
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    pip install --no-cache-dir torch-geometric

# Install remaining requirements (excluding PyTorch and PyG packages)
RUN grep -v "^torch" requirements.txt | grep -v "^--" > requirements-core.txt && \
    pip install --no-cache-dir -r requirements-core.txt && \
    rm requirements-core.txt

# Copy the rest of the application code
COPY . .

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]
