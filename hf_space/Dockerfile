# Use Python 3.11 slim as base (smaller image, faster builds)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Step 1: Install PyTorch CPU-only FIRST (separate layer for caching) ──────
# CPU-only wheels are ~750 MB smaller than the default CUDA build.
# This layer is cached as long as the torch version doesn't change.
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install remaining Python dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 3: Copy application code ─────────────────────────────────────────────
COPY backend/ ./backend/

# ── Step 4: Copy model files (tracked via Git LFS) ───────────────────────────
COPY models/ ./models/

# ── Hugging Face Spaces requires port 7860 ────────────────────────────────────
EXPOSE 7860

# Run the FastAPI app as a Python package so relative imports work correctly.
# backend/main.py uses `from .model_loader import ...` (relative imports).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
