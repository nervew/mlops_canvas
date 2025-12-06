import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from .azure_blob import BlobDownloader

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    path: str
    format: str


class BaseModelRunner:
    def predict(self, df: pd.DataFrame) -> List[Any]:
        raise NotImplementedError


class PickleModelRunner(BaseModelRunner):
    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        self.model = joblib.load(artifact.path)

    def predict(self, df: pd.DataFrame) -> List[Any]:
        logger.debug("Running pickle inference on %s rows", len(df))
        predictions = self.model.predict(df)
        return predictions.tolist()


class OnnxModelRunner(BaseModelRunner):
    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        self.session = ort.InferenceSession(artifact.path, providers=["CPUExecutionProvider"])

    def predict(self, df: pd.DataFrame) -> List[Any]:
        logger.debug("Running ONNX inference on %s rows", len(df))
        inputs = self._build_inputs(df)
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs)
        # if single output, flatten
        if len(outputs) == 1:
            return np.ravel(outputs[0]).tolist()
        return [list(map(float, row)) for row in zip(*outputs)]

    def _build_inputs(self, df: pd.DataFrame) -> Dict[str, Any]:
        # align with ONNX input names; default to one input using all values
        session_inputs = self.session.get_inputs()
        if len(session_inputs) == 1:
            input_name = session_inputs[0].name
            return {input_name: df.to_numpy().astype(np.float32)}
        inputs: Dict[str, Any] = {}
        for input_meta in session_inputs:
            col = input_meta.name
            if col in df.columns:
                inputs[col] = df[col].to_numpy().astype(np.float32)
            else:
                raise ValueError(f"Missing required input column: {col}")
        return inputs


def load_model(downloader: BlobDownloader, blob_path: str) -> BaseModelRunner:
    _, ext = os.path.splitext(blob_path.lower())
    temp_path = downloader.download_to_temp(blob_path)
    artifact = ModelArtifact(path=temp_path, format=ext)
    logger.info("Model downloaded to %s with format %s", temp_path, ext)

    if ext in {".pkl", ".pickle"}:
        return PickleModelRunner(artifact)
    if ext == ".onnx":
        return OnnxModelRunner(artifact)
    raise ValueError(f"Unsupported model format: {ext}")


__all__ = ["load_model", "PickleModelRunner", "OnnxModelRunner"]
