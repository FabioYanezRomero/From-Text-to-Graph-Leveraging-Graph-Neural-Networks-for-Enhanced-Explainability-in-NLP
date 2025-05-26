# Base image
FROM python:3.8-slim as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0 \
    TORCH_VERSION=1.12.1+cu113 \
    TORCHVISION_VERSION=0.13.1+cu113 \
    TORCHAUDIO_VERSION=0.12.1 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    procps \
    lsb-release \
    python3-dev \
    python3-pip \
    cmake \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && pip install --no-cache-dir pybind11 \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up to date
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with pre-built wheels
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    nltk \
    tqdm \
    && pip install --no-cache-dir poetry==$POETRY_VERSION \
    && poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false

# Install additional build dependencies for binary wheels
RUN pip install --no-cache-dir --only-binary=:all: \
    cython>=0.29.0 \
    pybind11>=2.10.0

RUN pip install --no-cache-dir \
    numpy>=1.23.5 \
    scipy>=1.10.1 \
    --only-binary=:all:

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install additional dependencies with pre-built wheels
RUN pip install --no-cache-dir --only-binary=:all: \
    transformers>=4.0.0 \
    stanza>=1.3.0 \
    nltk \
    tqdm \
    matplotlib \
    seaborn

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas64-0 \
    && rm -rf /var/lib/apt/lists/*

# Keep build dependencies for now, we'll clean them up after installing AutoGOAL
RUN mkdir -p /app
WORKDIR /app

# Development stage
FROM base as development

# Install development system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    git-lfs \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages with pre-built wheels
RUN pip install --no-cache-dir --only-binary=:all: \
    ipython \
    ipdb \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    jupyterlab \
    notebook \
    numpy==1.23.5 \
    scipy==1.10.1 \
    setuptools==59.5.0

# Copy the application code
COPY . .

# Install build dependencies for scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install AutoGOAL in development mode if it exists
RUN if [ -d "autogoal_repo/autogoal" ]; then \
        # First install numpy and scipy with specific versions
        pip install --no-cache-dir --only-binary=:all: numpy==1.20.3 && \
        pip install --no-cache-dir scipy==1.6.0 && \
        # Then install AutoGOAL in development mode
        cd autogoal_repo/autogoal && \
        pip install --no-cache-dir -e . && \
        cd /app; \
    fi

# Clean up build dependencies to reduce image size
RUN apt-get remove -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip/*

# Install project dependencies if pyproject.toml exists
RUN if [ -f "pyproject.toml" ]; then \
        poetry install --no-interaction --no-ansi; \
    fi

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command to keep the container running
CMD ["bash"]

# Production stage
FROM base as production

# Install production system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install numpy and scipy with pre-built binaries
RUN pip install --no-cache-dir \
    --only-binary=:all: \
    numpy>=1.23.5 \
    scipy>=1.10.1

# Install AutoGOAL in production mode if it exists
RUN if [ -d "autogoal_repo/autogoal" ]; then \
        cd autogoal_repo/autogoal && \
        pip install --no-cache-dir -e . --no-deps && \
        cd /app; \
    fi

# Install project dependencies if pyproject.toml exists
RUN if [ -f "pyproject.toml" ]; then \
        poetry install --no-interaction --no-ansi --no-dev; \
    fi

# Clean up to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory to src where the main code is located
WORKDIR /app/src

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["python", "main.py"]


