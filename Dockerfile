FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code + model files
COPY models/ ./models/
COPY predict.py .
COPY catboost_model.pkl .

COPY deberta_v2_best.pt .

# cache DeBERTa tokenizer + model weights at build time so inference is fully offline
RUN python -c "\
import torch; \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large'); \
AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=1, torch_dtype=torch.float32); \
print('DeBERTa cached')"

ENTRYPOINT ["python", "predict.py"]
