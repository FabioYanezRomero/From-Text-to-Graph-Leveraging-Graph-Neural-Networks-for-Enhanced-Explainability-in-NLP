# Dedicated Dockerfile for GraphSVX explanations
FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    libopenmpi-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set workdir
WORKDIR /graphsvx

# Clone GraphSVX repository
RUN git clone https://github.com/AlexDuvalinho/GraphSVX.git .

# Upgrade pip and install PyTorch and dependencies (CUDA 12.1 version for GPU support)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric and dependencies (CUDA 12.1 wheels)
RUN python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    python3 -m pip install torch-geometric

# Install other Python dependencies for GraphSVX
RUN python3 -m pip install -r requirements.txt

# Copy and install application requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Optional: install Jupyter for interactive use
RUN python3 -m pip install jupyter

# Entrypoint for running GraphSVX scripts
ENTRYPOINT ["/bin/bash"]
