# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set environment variables
ARG BUILD_ENVIRONMENT=development
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUILD_ENV=${BUILD_ENVIRONMENT} \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/home/coder/autogoal:/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create coder user and set up directories
RUN useradd -m coder && \
    mkdir -p /home/coder/autogoal && \
    chown -R coder:coder /home/coder

# Set working directory
WORKDIR /home/coder/autogoal

# Copy the AutoGOAL repository
COPY --chown=coder:coder ./autogoal_repo .

# Switch to coder user
USER coder

# Initialize and update git submodules
RUN git submodule update --init --recursive

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

# Install AutoGOAL core
RUN cd /home/coder/autogoal/autogoal && \
    pip install -e .

# Install autogoal-contrib if it exists
RUN if [ -d "/home/coder/autogoal/autogoal-contrib" ]; then \
        cd /home/coder/autogoal/autogoal-contrib && \
        pip install -e .; \
    fi

# Install autogoal-remote if it exists
RUN if [ -d "/home/coder/autogoal/autogoal-remote" ]; then \
        cd /home/coder/autogoal/autogoal-remote && \
        pip install -e .; \
    fi

# Set the default command
CMD ["/bin/bash"]
