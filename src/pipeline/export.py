import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from .constants import MODEL_DIR

def create_model(grid, X_train, y_train, version: str = "v1"):
    model = grid.best_estimator_
    model.fit(X_train, y_train)

    onnx_path   = MODEL_DIR / f"model_{version}.onnx"
    joblib_path = MODEL_DIR / f"model_{version}.joblib"

    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model   = convert_sklearn(model, initial_types=initial_type)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    joblib.dump(model, joblib_path)

    print(f"✅ Modelo guardado:\n   • {joblib_path}\n   • {onnx_path}", flush=True)
    return model
