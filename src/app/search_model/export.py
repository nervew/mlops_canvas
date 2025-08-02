from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# NUEVOS imports
import lightgbm
from onnxmltools import convert_lightgbm        # <- conversor propio

def _convert_lightgbm(model, X_sample):
    """Convierte un modelo LightGBM a ONNX."""
    return convert_lightgbm(
        model, 
        initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))]
    )

def export_model_onnx(model, X_sample, output_dir, version="v1"):
    """
    Exporta un modelo a ONNX.
    Detecta LightGBM automáticamente y usa el conversor adecuado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"modelo_{version}.onnx"

    try:
        # --- detección de LightGBM -----------------------------------------
        if isinstance(model, (lightgbm.LGBMRegressor, lightgbm.LGBMClassifier)):
            onx = _convert_lightgbm(model, X_sample)
        else:
            onx = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))]
            )
        # -------------------------------------------------------------------
        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"✔ Modelo ONNX guardado en {filename}", flush=True)
    except Exception as exc:
        print(f"⚠ Error exportando ONNX: {exc}", flush=True)
