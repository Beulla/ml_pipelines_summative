# API image: FastAPI + PyTorch (CPU). Serves /predict, /upload-data, /retrain, /evaluate.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image ~1 GB instead of ~5 GB with CUDA.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY models ./models
COPY data/train ./data/train
COPY data/test ./data/test

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:${PORT}/health || exit 1
# Render/Hugging Face inject $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
