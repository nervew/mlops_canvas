"""FastAPI application for runtime inference in Azure Container Apps."""

from __future__ import annotations

import io
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from azure.storage.blob import BlobServiceClient
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Azure MLOps Inference API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request payload for inference."""

    model_name: str = Field(
        ..., description="Nombre lógico del modelo en Blob Storage (prefijo de carpeta)."
    )
    model_blob_path: Optional[str] = Field(
        None,
        description=(
            "Ruta relativa del modelo dentro del contenedor. Si no se envía, se usa"
            " `<model_name>/model/<model_name>.pickle`."
        ),
    )
    records: Optional[List[Dict[str, Any]]] = Field(
        None, description="Datos tabulares en formato JSON para inferencia."
    )
    file_format: Optional[str] = Field(
        None,
        regex="^(csv|parquet|pickle)$",
        description="Formato del archivo adjunto cuando se usa carga por archivo.",
    )

    @validator("records")
    def validate_records(cls, value: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if value is not None and len(value) == 0:
            raise ValueError("records no puede ser una lista vacía")
        return value


@app.get("/health", summary="Healthcheck")
async def health() -> Dict[str, str]:
    """Endpoint de salud simple para Container Apps."""

    return {"status": "ok"}


def _blob_client() -> Tuple[str, str, BlobServiceClient]:
    """Crea un cliente de contenedor de blobs usando variables de entorno."""

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_MODEL_CONTAINER", "models")

    if not connection_string:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING es obligatorio para descargar modelos"
        )

    service_client = BlobServiceClient.from_connection_string(connection_string)
    return container_name, connection_string, service_client


def _download_model_from_blob(blob_path: str) -> Path:
    """Descarga el modelo desde Blob Storage a un archivo temporal."""

    container_name, _, service_client = _blob_client()
    container_client = service_client.get_container_client(container_name)

    if not container_client.exists():
        raise HTTPException(
            status_code=404,
            detail=f"El contenedor {container_name} no existe en la cuenta de almacenamiento",
        )

    blob_client = container_client.get_blob_client(blob=blob_path)
    if not blob_client.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el modelo en la ruta {blob_path}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(blob_path).suffix) as temp_file:
        download_stream = blob_client.download_blob()
        download_stream.readinto(temp_file)
        logger.info("Modelo descargado temporalmente en %s", temp_file.name)
        return Path(temp_file.name)


def _load_model(local_path: Path) -> Tuple[Any, str]:
    """Carga modelos pickle u ONNX desde un archivo local."""

    if local_path.suffix in {".pkl", ".pickle"}:
        with open(local_path, "rb") as handle:
            model = pickle.load(handle)
            return model, "pickle"
    if local_path.suffix == ".onnx":
        session = ort.InferenceSession(local_path.as_posix())
        return session, "onnx"

    raise HTTPException(
        status_code=400,
        detail=(
            "Formato de modelo no soportado. Solo se aceptan archivos .pickle, .pkl u .onnx"
        ),
    )


def _dataframe_from_payload(
    payload: PredictionRequest, upload_file: Optional[UploadFile]
) -> pd.DataFrame:
    """Genera un DataFrame a partir de JSON o archivos cargados."""

    if upload_file:
        file_bytes = upload_file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="El archivo adjunto está vacío")

        file_format = payload.file_format or upload_file.filename.split(".")[-1]
        buffer = io.BytesIO(file_bytes)
        if file_format == "csv":
            df = pd.read_csv(buffer)
        elif file_format == "parquet":
            df = pd.read_parquet(buffer)
        elif file_format == "pickle":
            df = pd.read_pickle(buffer)
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato de archivo no soportado. Use csv, parquet o pickle",
            )
        return df

    if payload.records:
        return pd.DataFrame(payload.records)

    raise HTTPException(
        status_code=400,
        detail="Debe proveerse `records` o un archivo para la inferencia",
    )


def _run_inference(model: Any, model_type: str, data: pd.DataFrame) -> List[Any]:
    """Ejecuta inferencia en función del tipo de modelo."""

    if data.empty:
        raise HTTPException(status_code=400, detail="Los datos de entrada están vacíos")

    if model_type == "pickle":
        if not hasattr(model, "predict"):
            raise HTTPException(
                status_code=400,
                detail="El objeto cargado no expone un método predict",
            )
        predictions = model.predict(data)
        return predictions.tolist()

    if model_type == "onnx":
        input_name = model.get_inputs()[0].name
        input_data = data.to_numpy().astype(np.float32)
        outputs = model.run(None, {input_name: input_data})
        return outputs[0].tolist()

    raise HTTPException(status_code=500, detail="Tipo de modelo desconocido")


@app.post("/infer", summary="Realiza inferencia usando un modelo en Blob Storage")
async def infer(  # type: ignore[override]
    payload: PredictionRequest = Body(...),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    """Descarga el modelo, valida los datos y devuelve las predicciones."""

    model_blob_path = payload.model_blob_path or f"{payload.model_name}/model/{payload.model_name}.pickle"

    try:
        data = _dataframe_from_payload(payload, file)
        local_model_path = _download_model_from_blob(model_blob_path)
        model, model_type = _load_model(local_model_path)
        predictions = _run_inference(model, model_type, data)
        logger.info(
            "Inferencia completada para modelo=%s tipo=%s filas=%s",
            payload.model_name,
            model_type,
            len(data),
        )
    except HTTPException as http_exc:
        logger.exception("Error en solicitud de inferencia")
        raise http_exc
    except Exception as exc:  # pragma: no cover - fallback
        logger.exception("Fallo inesperado durante la inferencia")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if "local_model_path" in locals():
            try:
                Path(local_model_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("No se pudo eliminar el archivo temporal %s", local_model_path)

    return JSONResponse(
        content={
            "model_name": payload.model_name,
            "model_blob_path": model_blob_path,
            "rows_received": len(data),
            "predictions": predictions,
        }
    )


def build_default_app() -> FastAPI:
    """Permite crear la app desde otras entradas (por ejemplo, gunicorn)."""

    return app


__all__ = ["app", "build_default_app", "PredictionRequest"]
