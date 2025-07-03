import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator


def export_model(model: BaseEstimator, X_shape: int, path: str = "model.onnx") -> None:
    onnx_model = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, X_shape]))])
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    joblib.dump(model, path.replace(".onnx", ".joblib"))
