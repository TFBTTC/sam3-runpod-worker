# SAM3 RunPod Serverless Worker
# Requirements: Python 3.12, PyTorch 2.7+, CUDA 12.6+
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install SAM3 dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /workspace/handler.py

ENV SAM3_MODEL_ID="facebook/sam3"
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE="/workspace/hf_cache"

# Pre-create cache dir
RUN mkdir -p /workspace/hf_cache

CMD ["python", "-u", "/workspace/handler.py"]
