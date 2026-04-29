# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY app.py .

# Create __init__.py files
RUN touch src/__init__.py api/__init__.py

# Expose ports
EXPOSE 8000 8501

# Default command (API)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]