# Use a stable, slim Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl/unzip needed for AWS CLI if you add it later)
# For now, we just need basic python tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Python path so 'app' module is found
ENV PYTHONPATH=/app

# Default command
CMD ["python", "app/train.py"]