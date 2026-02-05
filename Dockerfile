# Use an official Python runtime as a parent image
# Python 3.10 is stable for InsightFace/ONNX
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies required for OpenCV and InsightFace
# libgl1-mesa-glx: Required for cv2
# libglib2.0-0: Required for some cv2 dependencies
# gcc, g++: Required for compiling some Python packages (like insightface extensions)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 5000
EXPOSE 5000

# Run with Gunicorn for Production
# Workers: 1 (Limited for Free Tier Ram) or 2 if swap enabled
# Timeout: 120s (InsightFace loading can be slow)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
