FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEBSITES_PORT=8080 \
    MODEL_DIR=/models

WORKDIR /home/site/wwwroot
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080
CMD ["python", "-m", "azure.functions.worker"]
