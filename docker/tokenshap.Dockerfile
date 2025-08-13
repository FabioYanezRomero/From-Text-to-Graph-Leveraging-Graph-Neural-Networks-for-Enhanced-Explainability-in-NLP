# Dedicated Dockerfile for TokenSHAP (token-level SHAP for LLMs) with GPU support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /tokenshap

# Upgrade pip/setuptools/wheel first
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.8 build) + torchvision + torchaudio
# Reference: https://pytorch.org
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchvision==0.23.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install Transformers, Datasets, and related NLP utilities
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    huggingface_hub \
    sentencepiece

# Install TokenSHAP from PyPI
RUN pip install --no-cache-dir tokenshap

# Optional: Jupyter for interactive development
RUN pip install --no-cache-dir jupyter

# Pre-create cache directories and set permissions
RUN mkdir -p /opt/hf_cache && chmod -R 777 /opt/hf_cache


ENTRYPOINT ["/bin/bash"]
