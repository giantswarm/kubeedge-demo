# Use a minimal Debian-based Python image
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    paho-mqtt

# Copy application code
COPY advanced-multi-person-counter.py /app/

# Create config directory for mounted config
RUN mkdir -p /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config/config.json

# Run the Python script
CMD ["python", "advanced-multi-person-counter.py"]