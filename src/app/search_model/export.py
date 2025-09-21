# src/app/search_model/export.py

from pathlib import Path
import onnx
from skl2onnx import convert_sklearn, __max_supported_opset__
from skl2onnx.common.data_types import FloatTensorType

import lightgbm
import xgboost
from onnxmltools import convert_lightgbm, convert_xgboost


# ------------------------------------------------------------------ #
def get_log_path(filename: str = "flaml.log") -> str:
    """
    Ruta absoluta a <root>/logs/<filename>, creando el directorio si hace falta.
    """
    root = Path(__file__).resolve().parent
    while (root / "__init__.py").exists():
        root = root.parent
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / filename)


def _effective_opset(desired=17, xgb_limit=15):
    """
    Devuelve opsets máximos para:
      - 'default': min(desired, onnx_max, skl2onnx_max)
      - 'xgboost': min(default, xgb_limit)
    """
    onnx_max = onnx.defs.onnx_opset_version()
    skl_max  = __max_supported_opset__
    common   = min(desired, onnx_max, skl_max)
    return {"default": common, "xgboost": min(common, xgb_limit)}


# ----------------------- helpers de conversión --------------------- #
def _unwrap_model(model):
    """
    Si el modelo viene envuelto por FLAML (p.ej. XGBoostSklearnEstimator, LGBMEstimator),
    devuelve el estimador real en `model.model`. En otro caso, devuelve el original.
    """
    try:
        mod = type(model).__module__
        if mod and mod.startswith("flaml.automl.model") and hasattr(model, "model"):
            inner = getattr(model, "model", None)
            if inner is not None:
                return inner
    except Exception:
        pass
    return model


def _convert_lightgbm(model, X_sample, opset):
    return convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))],
        target_opset=opset,
    )


def _convert_xgboost(model, X_sample, opset):
    """
    Convierte XGBRegressor/Classifier a ONNX.
    • Forzamos opset ≤ 15 (límite del conversor).
    • Renombramos internamente booster.feature_names a f0, f1, … 
      para evitar errores de nombre y de count.
    """
    booster = model.get_booster()

    # 1) Intentar extraer número real de features del wrapper sklearn
    n_feats = getattr(model, "n_features_in_", None)
    if n_feats is None:
        # 2) Caída a len(booster.feature_names) si existe
        fn = getattr(booster, "feature_names", None)
        n_feats = len(fn) if fn is not None else X_sample.shape[1]

    # 3) Renombrar exactamente n_feats nombres
    booster.feature_names = [f"f{i}" for i in range(n_feats)]

    return convert_xgboost(
        booster,
        initial_types=[("input", FloatTensorType([None, n_feats]))],
        target_opset=min(opset, 15),
    )


# --------------------------- API pública --------------------------- #
def export_model_onnx(model, X_sample, output_dir, version="v1"):
    """
    Exporta un modelo (sklearn, LightGBM o XGBoost) a ONNX,
    eligiendo el conversor adecuado y ajustando opset/feature_names.
    Acepta tanto estimadores "puros" como wrappers de FLAML (se desenvuelven).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"modelo_{version}.onnx"

    # Si es un wrapper de FLAML, obtener el estimador real
    model = _unwrap_model(model)

    opsets = _effective_opset()

    try:
        # Selección del conversor según tipo de modelo real
        if isinstance(model, (lightgbm.LGBMRegressor, lightgbm.LGBMClassifier)):
            onx = _convert_lightgbm(model, X_sample, opsets["default"])
            used = opsets["default"]
        elif isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
            onx = _convert_xgboost(model, X_sample, opsets["xgboost"])
            used = opsets["xgboost"]
        else:
            # Para modelos sklearn compatibles directamente con skl2onnx
            onx = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))],
                target_opset=opsets["default"],
            )
            used = opsets["default"]

        # Guardar el archivo
        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"✔ Modelo ONNX guardado en {filename} (opset {used})", flush=True)

    except Exception as exc:
        print(f"⚠ Error exportando ONNX: {exc}", flush=True)
        print("  Sugerencia: si el mejor estimador es CatBoost u otro no soportado por skl2onnx/onnxmltools, "
              "usa un estimador sklearn/LGBM/XGBoost o ajusta la exportación.", flush=True)
