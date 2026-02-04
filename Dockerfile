# Base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency definitions first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv sync --no-dev

# Copy application source code
COPY src/ ./src/
COPY models/ ./models/
COPY logs/ ./logs/

# Expose API port
EXPOSE 5000

# Run the application
CMD ["uv", "run", "python", "-m", "api_breast_cancer_prediction"]
