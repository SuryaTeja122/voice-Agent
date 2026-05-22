FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing + torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

Run pip install --upgrade pip setuptools

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Download Silero VAD model weights
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=True)" || true

# Download fastText language ID model
RUN curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
    -o /app/models/lid.176.ftz 2>/dev/null || mkdir -p /app/models

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
