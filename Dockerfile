# ==============================================================================
# Neuro CXR Diagnostic API - Production Dockerfile for Render Deployment
# Maintainer: Debashis Mohapatra <debashismohapatra8260@gmail.com>
# Year: 2026
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Non-interactive installation and unbuffered stream logging
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and torchvision for lightweight deployment
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Pin NumPy 1.x and OpenCV headless to guarantee ABI stability
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84"

# Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY backend/ ./backend/
COPY models/ ./models/

# Expose container port
EXPOSE 8000

# Start Uvicorn dynamically binding to Render's $PORT environment variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
