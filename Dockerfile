FROM python:3.11-slim

WORKDIR /app

COPY artifacts/requirements.txt artifacts/requirements.txt
RUN pip install --no-cache-dir -r artifacts/requirements.txt

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY artifacts/ artifacts/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
