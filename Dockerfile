# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Set specific versions for compatibility
    TORCH_VERSION=1.12.1+cu113 \
    TORCHVISION_VERSION=0.13.1+cu113 \
    TORCHAUDIO_VERSION=0.12.1 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    libopenblas-dev \
    libsndfile1 \
    libboost-all-dev \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    # Additional dependencies for PyTorch and Supar
    libopenmpi-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install system-level Python packages first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (required by Supar)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install Poetry and configure
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && poetry config virtualenvs.create false

# Install additional system-level dependencies
RUN pip install --no-cache-dir \
    numpy>=1.23.5 \
    scipy>=1.10.1 \
    cython>=0.29.0 \
    transformers>=4.0.0 \
    stanza>=1.3.0

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# Copy the rest of the application
COPY . .

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]
