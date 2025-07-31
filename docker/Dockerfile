# Use NVIDIA CUDA 12.1 with Ubuntu 22.04 as base image for better PyTorch Geometric support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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
    libopenblas-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and its dependencies with specific versions for CUDA 12.1
RUN pip install --no-cache-dir \
    pyg_lib==0.3.0+pt21cu121 \
    torch_scatter==2.1.2+pt21cu121 \
    torch_sparse==0.6.18+pt21cu121 \
    torch_cluster==1.6.3+pt21cu121 \
    torch_spline_conv==1.2.2+pt21cu121 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch_geometric==2.4.0

# Install remaining dependencies from requirements.txt, excluding PyTorch and PyTorch Geometric related packages
COPY requirements.txt .
RUN grep -v -E 'torch|pytorch_geometric|pyg_lib|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv' requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements.txt requirements_filtered.txt

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
