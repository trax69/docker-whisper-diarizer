# USAMOS LA BASE CUDA 11.8 (Compatible con tus logs)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Remove NVIDIA apt source to avoid mirror sync failures (CUDA is already in the base image)
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

COPY src/ /app/src/
COPY transcribir.py /app/transcribir.py

RUN mkdir -p /app/data/input /app/data/output /app/data/processing /app/data/completed

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["python", "src/main.py"]