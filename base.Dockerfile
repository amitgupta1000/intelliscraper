# base.Dockerfile
# This Dockerfile creates a custom base image with all heavy dependencies pre-installed.
# Build this image and push it to your container registry.
# Example: docker build -f base.Dockerfile -t your-registry/deepsearch-base:1.0 .

# --- STAGE 1: Builder ---
# Use a full image with build tools to compile dependencies into wheels.
FROM python:3.12-bullseye AS builder

WORKDIR /app

# Define a shared location for the HuggingFace model cache
ENV HF_HOME=/opt/huggingface_cache

# Install build tools and create the wheelhouse
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip wheel \
    && pip wheel --no-cache-dir -r requirements.txt -w /wheels \
    # Install dependencies into the builder to run the download script
    && pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Download and cache the sentence-transformer model within the builder stage
COPY backend/src/download_model.py backend/src/config.py backend/src/logging_setup.py ./backend/src/
RUN python -m backend.src.download_model

# --- STAGE 2: Final Base Image ---
# Start from a slim image for the final base.
FROM python:3.12-slim-bullseye

# Set environment variables that will be inherited by the final application image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    HF_HOME=/opt/huggingface_cache \
    GUNICORN_TIMEOUT=60

WORKDIR /app

# Install only the necessary runtime system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libdbus-1-3 dbus libatk1.0-0 libatk-bridge2.0-0 libcups2 libatspi2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    libasound2 libfontconfig1 libxkbcommon0 xkb-data curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from the builder stage
COPY --from=builder /wheels /wheels
COPY --from=builder $HF_HOME $HF_HOME
COPY requirements.txt ./
RUN pip install --no-index --no-cache-dir --find-links /wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Install Playwright system dependencies so browsers run at runtime
RUN playwright install-deps || true

# Install the Playwright browser binaries
RUN playwright install chromium

# Create the non-root user that the final application will use
RUN useradd -m -s /bin/bash appuser

# Set correct ownership for the app directory and the caches.
# This ensures the non-root user can access everything it needs.
RUN chown -R appuser:appuser /app $PLAYWRIGHT_BROWSERS_PATH $HF_HOME