FROM python:3.11-slim

ARG PIP_EXTRA_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies
COPY infra/azure_inference_service/runtime_requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Application code
COPY infra/azure_inference_service /app/infra/azure_inference_service

EXPOSE 8080
CMD ["uvicorn", "infra.azure_inference_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
