# Stage 1 — install and compile all dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all packages; fastembed uses ONNX (no PyTorch/CUDA needed)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the ONNX embedding model into the image so startup is instant
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Stage 2 — lean runtime image (no compiler toolchain)
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /root/.cache/fastembed /root/.cache/fastembed

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
