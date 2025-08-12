# Use a minimal Python image
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir paho-mqtt

# Copy application code
COPY mock-hand-counter.py /app/

# Create config directory for mounted config
RUN mkdir -p /app/config

# Note: config.json will be provided by ConfigMap at runtime
# mounted to /app/config/config.json

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config/config.json

# Add a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Run the Python script
CMD ["python", "mock-hand-counter.py"]
