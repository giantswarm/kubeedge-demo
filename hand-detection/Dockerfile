# Use a minimal Debian-based Python image
FROM python:3.12.9-slim

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless

# Set working directory
WORKDIR /app

# Copy application code
COPY raise_hand.py /app/

# Run the Python script
CMD ["python", "raise_hand.py"]