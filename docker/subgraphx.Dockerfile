# Docker image dedicated to running SubgraphX explanations with DIG support.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /subgraphx

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install "numpy<2"

RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    pyg_lib==0.4.0+pt23cu121 \
    torch_scatter==2.1.2+pt23cu121 \
    torch_sparse==0.6.18+pt23cu121 \
    torch_cluster==1.6.3+pt23cu121 \
    torch_spline_conv==1.2.2+pt23cu121 \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

RUN pip install --no-cache-dir torch_geometric==2.5.3

RUN pip install --no-cache-dir rdkit-pypi==2023.9.6

RUN pip install --no-cache-dir dive-into-graphs

COPY requirements.txt /tmp/requirements.txt
RUN grep -v -E 'torch|pytorch_geometric|pyg_lib|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv|numpy' /tmp/requirements.txt > /tmp/requirements-subgraphx.txt && \
    pip install --no-cache-dir -r /tmp/requirements-subgraphx.txt && \
    rm /tmp/requirements.txt /tmp/requirements-subgraphx.txt

RUN pip install --no-cache-dir "numpy<2"

ENTRYPOINT ["/bin/bash"]
