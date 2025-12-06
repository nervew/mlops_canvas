from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator


class InferenceRequest(BaseModel):
    model_blob_path: Optional[str] = Field(
        None,
        description="Relative path in the blob container. Defaults to model_name/model/model.pkl",
    )
    model_name: Optional[str] = Field(None, description="Overrides default MODEL_NAME.")
    data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Records for inference. Provide list of JSON objects."
    )

    @root_validator
    def validate_payload(cls, values: Dict[str, Any]):
        data = values.get("data")
        if data is not None and not isinstance(data, list):
            raise ValueError("data must be a list of JSON objects")
        return values


class InferenceResponse(BaseModel):
    model_path: str
    predictions: List[Any]
    prediction_count: int


class HealthResponse(BaseModel):
    status: str


__all__ = ["InferenceRequest", "InferenceResponse", "HealthResponse"]
