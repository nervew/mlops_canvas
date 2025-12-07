# Imagen base ligera para FastAPI y dependencias de ML
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos requirements combinados (base + específicos descargados)
COPY build/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.lock

# Copiamos el código fuente de la API
COPY src/inference_api /app/inference_api

ENV HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["uvicorn", "inference_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
