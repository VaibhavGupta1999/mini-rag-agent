# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Speed up and keep image small
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for FAISS / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends     gcc g++ git &&     rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build a tiny sample index (optional, safe if it fails)
RUN python app/index/build_index.py --src data --out app/index/store || true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
