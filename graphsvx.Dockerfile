# Dedicated Dockerfile for GraphSVX explanations
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /graphsvx

# Clone GraphSVX repository
RUN git clone https://github.com/AlexDuvalinho/GraphSVX.git .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 \
    && pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    && pip install -r requirements.txt

# Optional: install Jupyter for interactive use
RUN pip install jupyter

# Entrypoint for running GraphSVX scripts
ENTRYPOINT ["/bin/bash"]
