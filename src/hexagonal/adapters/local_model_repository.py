from pathlib import Path
from typing import Any
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ..ports.model_repository import ModelRepository


class LocalModelRepository(ModelRepository):
    """Saves and loads models locally in ONNX and joblib formats."""

    def __init__(self, directory: str = "models") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = self.dir / "model.onnx"
        self.joblib_path = self.dir / "model.joblib"

    def save(self, model: Any) -> None:
        initial_type = [("input", FloatTensorType([None, model.n_features_in_]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(self.onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        joblib.dump(model, self.joblib_path)

    def load(self) -> Any:
        return joblib.load(self.joblib_path)
