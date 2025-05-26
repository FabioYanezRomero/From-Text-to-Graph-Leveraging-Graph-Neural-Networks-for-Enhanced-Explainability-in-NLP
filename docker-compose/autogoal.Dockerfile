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

# Install git as root
RUN apt-get update && apt-get install -y --no-install-recommends git

# Configure git to trust the directory
RUN git config --global --add safe.directory /home/coder/autogoal

# Clone your forked repository
RUN git clone --recursive https://github.com/FabioYanezRomero/autogoal_graph_explainability.git /home/coder/autogoal && \
    cd /home/coder/autogoal && \
    git remote add upstream https://github.com/autogoal/autogoal.git && \
    # Configure safe directories for submodules if they exist
    if [ -d "/home/coder/autogoal/autogoal-contrib" ]; then \
        git config --global --add safe.directory /home/coder/autogoal/autogoal-contrib; \
    fi && \
    if [ -d "/home/coder/autogoal/autogoal-remote" ]; then \
        git config --global --add safe.directory /home/coder/autogoal/autogoal-remote; \
    fi

# Fix permissions
RUN chown -R coder:coder /home/coder/autogoal

# Switch to coder user
USER coder

# If there are local changes in autogoal_repo, copy them selectively
# First, create a temporary directory for local changes
RUN mkdir -p /tmp/local_changes

# Copy local changes to the temporary directory
COPY --chown=coder:coder ./autogoal_repo/ /tmp/local_changes/

# Copy only the necessary files from local changes, excluding .git directory
RUN if [ -d "/tmp/local_changes" ]; then \
        rsync -a --exclude='.git' /tmp/local_changes/ /home/coder/autogoal/; \
        rm -rf /tmp/local_changes; \
    fi

# Set working directory
WORKDIR /home/coder/autogoal

# Install Python dependencies
ENV PATH="/home/coder/.local/bin:${PATH}"
RUN pip install --upgrade pip && \
    pip install --user poetry && \
    poetry --version && \
    poetry config virtualenvs.create false

# Install AutoGOAL core
RUN cd /home/coder/autogoal/autogoal && \
    pip install -e .

# Install autogoal-contrib if it exists and has a setup.py or pyproject.toml
RUN if [ -f "/home/coder/autogoal/autogoal-contrib/setup.py" ] || [ -f "/home/coder/autogoal/autogoal-contrib/pyproject.toml" ]; then \
        cd /home/coder/autogoal/autogoal-contrib && \
        pip install -e .; \
    elif [ -d "/home/coder/autogoal/autogoal-contrib" ]; then \
        echo "autogoal-contrib found but not installed (not a Python package)"; \
    fi

# Skip autogoal-remote installation as it's not required for basic functionality
# and is causing build issues due to missing README.md
RUN if [ -d "/home/coder/autogoal/autogoal-remote" ]; then \
        echo "Skipping autogoal-remote installation - not required for basic functionality"; \
    fi

# Set the default command
CMD ["/bin/bash"]
