# Dedicated Dockerfile for TokenSHAP (token-level SHAP for LLMs)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /tokenshap

# ============================
# OPTION 1: Install from PyPI
# ============================
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install tokenshap

# (If PyPI install is not available in your region, or you want the bleeding edge, use Option 2 instead.)

# ============================
# # OPTION 2: Install from source (most up-to-date)
# ============================
# RUN git clone https://github.com/ronigold/TokenSHAP.git .
# RUN python3 -m pip install -r requirements.txt

# Optional: install Jupyter for interactive development
RUN python3 -m pip install jupyter

ENTRYPOINT ["/bin/bash"]
