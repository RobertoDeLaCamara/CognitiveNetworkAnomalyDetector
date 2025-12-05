# Use a specific Python version for security and reproducibility
FROM python:3.11-slim

# Install security updates and required system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libpcap-dev \
        gcc \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r anomaly && useradd -r -g anomaly -s /bin/false anomaly

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with security considerations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Copy the project files into the container
COPY --chown=anomaly:anomaly . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data/training /app/data/test && \
    chown -R anomaly:anomaly /app && \
    chmod -R 755 /app

# Switch to non-root user
USER anomaly

# Set environment variables for security
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command to run the script
CMD ["python", "main.py"]