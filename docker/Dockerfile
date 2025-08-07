# Use NVIDIA CUDA 12.4.1 as base image
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid

# Add LLVM repository for OpenMP packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update

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
    libomp-14-dev \
    libomp-14-doc \
    libomp5-14 \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric with dependencies for PyTorch 2.6.0
RUN pip install --no-cache-dir \
    torch_geometric==2.5.0 \
    pyg_lib==0.4.0+pt26cu124 \
    torch_scatter==2.1.2+pt26cu124 \
    torch_sparse==0.6.18+pt26cu124 \
    torch_cluster==1.6.3+pt26cu124 \
    torch_spline_conv==1.2.2+pt26cu124 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Install remaining dependencies (excluding torch and geometric packages)
COPY requirements.txt .
RUN grep -v -E 'torch|pytorch_geometric|pyg_lib|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv' requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements.txt requirements_filtered.txt

# Copy the rest of the application code
COPY . .

# Set working directory for main code
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]
