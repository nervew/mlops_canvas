"""FastAPI application for online inference."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)

from .inference import InferenceService
from .model import load_model
from .schema import PredictionRequest, PredictionResponse

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO)

REGISTRY = CollectorRegistry(auto_describe=True)
LATENCY = Gauge(
    "prediction_latency_seconds",
    "Prediction latency",
    registry=REGISTRY
)
ERRORS = Gauge(
    "prediction_errors",
    "Prediction errors",
    registry=REGISTRY
)


@asynccontextmanager
def lifespan(_: FastAPI) -> AsyncIterator[None]:
    InferenceService.initialize(load_model())
    yield
    InferenceService.shutdown()


app = FastAPI(title="mlops-canvas", version="1.0.0", lifespan=lifespan)


def get_service() -> InferenceService:
    service = InferenceService.instance()
    if service is None:
        raise RuntimeError("Service not initialized")
    return service


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    response: Response = await call_next(request)
    request_id = request.headers.get("X-Request-Id", "unknown")
    response.headers["X-Request-Id"] = request_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    service: InferenceService = Depends(get_service)
) -> PredictionResponse:
    try:
        with LATENCY.time():
            result = service.predict(payload)
    except ValueError as exc:
        ERRORS.inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        ) from exc
    return result


@app.get("/metrics")
async def metrics() -> Response:
    payload = generate_latest(REGISTRY)
    return Response(payload, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
