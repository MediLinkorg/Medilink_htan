# CUDA 12.1 runtime — match to your EC2 GPU driver (Tesla T4 / g4dn is fine here).
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /srv

COPY requirements.txt .
# Torch CUDA wheels come from the PyTorch index.
RUN pip install --upgrade pip \
 && pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install -r requirements.txt

COPY models ./models
COPY app ./app
COPY scripts ./scripts

# Optional: bake the default model into the image (uncomment + pass HF_TOKEN if private)
# RUN python scripts/download_weights.py

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
