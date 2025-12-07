FROM python:3.11-slim

WORKDIR /app

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

COPY artifacts/requirements.txt artifacts/requirements.txt
RUN pip install --no-cache-dir -r artifacts/requirements.txt

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY artifacts/ artifacts/
COPY config.py .
COPY app.py .

EXPOSE ${API_PORT}

CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST} --port ${API_PORT}"]
