FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config.yaml .

EXPOSE 8000

# Secrets come from environment (docker-compose injects .env values); the
# ~2.2 GB models are bind-mounted from ./models (never baked into the image).
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
