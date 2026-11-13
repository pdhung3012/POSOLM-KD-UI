# GPU image by default. For CPU use --build-arg BASE_IMAGE=python:3.11-slim
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install Python dependencies only; Hugging Face model weights are mounted or cached at runtime.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir 'gunicorn>=22,<24'

COPY server.py model_loader.py /app/
COPY config/ /app/config/
COPY ui/ /app/ui/

EXPOSE 5000

# Exactly one worker: each process would otherwise load another multi-GB LLM.
# Threaded requests are protected by server.py's inference lock.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "600", "server:app"]
