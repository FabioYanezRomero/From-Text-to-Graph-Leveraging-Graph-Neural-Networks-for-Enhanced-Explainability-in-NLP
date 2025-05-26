# Use the official AutoGOAL image as base
FROM autogoal/autogoal:latest

# Set environment variables
ARG BUILD_ENVIRONMENT=development
ARG EXTRAS=all
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUILD_ENV=${BUILD_ENVIRONMENT} \
    POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /autogoal_repo

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

# Copy only the necessary files for dependency installation
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
RUN if [ -f pyproject.toml ]; then \
        poetry install --no-interaction --no-ansi $(test "$BUILD_ENV" = "production" && echo "--no-dev"); \
    fi

# Install AutoGOAL with specified extras
RUN if [ "$EXTRAS" != "none" ]; then \
        pip install autogoal[$EXTRAS]; \
    fi

# Copy the rest of the application
COPY . .

# Set the default command
CMD ["/bin/bash"]
