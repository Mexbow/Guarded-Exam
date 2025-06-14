# Base image with GPU support and pre-installed PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (only what's necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*
   
# Install pip
RUN apt-get update && apt-get install -y python3-pip
  
# Copy requirements and install
COPY requirements.txt .
RUN pip install --default-timeout=200 --prefer-binary --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8000

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]

