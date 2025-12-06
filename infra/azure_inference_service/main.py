import io
import logging
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .azure_blob import BlobDownloader
from .config import get_settings
from .model_runner import load_model
from .schemas import HealthResponse, InferenceRequest, InferenceResponse

logger = logging.getLogger("azure_inference_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Azure Model Inference", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _infer_blob_path(payload: InferenceRequest) -> str:
    settings = get_settings()
    model_name = payload.model_name or settings.model_name
    if payload.model_blob_path:
        return payload.model_blob_path
    return f"{model_name}/model/{model_name}.pickle"


def _dataframe_from_payload(payload: InferenceRequest, file: Optional[UploadFile]) -> pd.DataFrame:
    if payload.data is None and file is None:
        raise HTTPException(status_code=400, detail="Provide data[] or upload a file")

    if file:
        content = awaitable_read(file)
        return _load_dataframe_from_file(file.filename, content)

    if not payload.data:
        raise HTTPException(status_code=400, detail="data must be a non-empty list")

    try:
        return pd.DataFrame(payload.data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid data format: {exc}") from exc


def _load_dataframe_from_file(filename: str, content: bytes) -> pd.DataFrame:
    lower_name = filename.lower()
    buffer = io.BytesIO(content)
    if lower_name.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower_name.endswith(".parquet"):
        return pd.read_parquet(buffer)
    if lower_name.endswith(".json"):
        return pd.read_json(buffer)
    raise HTTPException(status_code=400, detail="Unsupported file type for inference upload")


def awaitable_read(upload: UploadFile) -> bytes:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return content


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    payload: InferenceRequest,
    file: Optional[UploadFile] = File(default=None),
    settings=Depends(get_settings),
):
    logger.info("Starting inference for model %s", payload.model_name or settings.model_name)
    try:
        df = _dataframe_from_payload(payload, file)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unable to parse input payload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    downloader = BlobDownloader(settings.azure_storage_connection_string, settings.model_blob_container)
    blob_path = _infer_blob_path(payload)
    try:
        runner = load_model(downloader, blob_path)
        predictions = runner.predict(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return InferenceResponse(model_path=blob_path, predictions=predictions, prediction_count=len(predictions))


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):  # type: ignore[override]
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
