# Use official lightweight Python base image
FROM python:3.11-slim

# Install system dependencies (FFmpeg + OpenCV requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install CPU-only PyTorch + Flask dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir flask opencv-python-headless numpy gunicorn

# Copy application files
COPY . .

# Create uploads and outputs directories
RUN mkdir -p uploads outputs

# Expose default port
EXPOSE 5000 10000

# Run single Gunicorn worker with 2 threads to stay under 512MB RAM on free hosting tiers
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 300 app:app"]
