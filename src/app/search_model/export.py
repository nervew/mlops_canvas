# search_model/export.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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


def _effective_opset(desired: int = 17, xgb_limit: int = 15) -> dict:
    """
    Devuelve opsets máximos:
      - 'default': min(desired, onnx_max, skl2onnx_max)
      - 'xgboost': min(default, xgb_limit)  (limitación del conversor)
    """
    onnx_max = onnx.defs.onnx_opset_version()
    skl_max = __max_supported_opset__
    common = min(desired, onnx_max, skl_max)
    return {"default": common, "xgboost": min(common, xgb_limit)}


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
      y calculamos el nº real de features usados por el booster/estimator
      para evitar mismatches de dimensión.
    """
    booster = model.get_booster()

    # 1) nº real de columnas que el wrapper cree que tiene
    n_feats = getattr(model, "n_features_in_", None)
    if n_feats is None:
        fn = getattr(booster, "feature_names", None)
        n_feats = len(fn) if fn is not None else X_sample.shape[1]

    # 2) Renombrar exactamente n_feats nombres
    booster.feature_names = [f"f{i}" for i in range(int(n_feats))]

    return convert_xgboost(
        booster,
        initial_types=[("input", FloatTensorType([None, int(n_feats)]))],
        target_opset=min(opset, 15),
    ), int(n_feats)


# ------------------------------------------------------------------ #
def export_model_onnx(
    model,
    X_sample,
    output_dir,
    version: str = "v1",
) -> Tuple[Optional[Path], Optional[int]]:
    """
    Exporta un modelo (sklearn, LightGBM o XGBoost) a ONNX.
    Devuelve (ruta_onnx | None, n_features_usadas | None).

    • Si el modelo no es convertible (p.ej. CatBoost), NO escribe nada
      y devuelve (None, None) para que el pipeline lo maneje con gracia.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"modelo_{version}.onnx"

    opsets = _effective_opset()

    try:
        used_n = int(X_sample.shape[1])

        if isinstance(model, (lightgbm.LGBMRegressor, lightgbm.LGBMClassifier)):
            onx = _convert_lightgbm(model, X_sample, opsets["default"])
        elif isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
            onx, used_n = _convert_xgboost(model, X_sample, opsets["xgboost"])
        else:
            # sklearn puro
            onx = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))],
                target_opset=opsets["default"],
            )

        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())

        print(f"✔ Modelo ONNX guardado en {filename} (input_dim={used_n})", flush=True)
        return filename, used_n

    except Exception as exc:
        # Modelos no soportados (ej. CatBoost) o fallos de conversión
        print(f"⚠ Error exportando ONNX: {exc}", flush=True)
        print("  → No se generó archivo .onnx (se omitirá la inferencia ONNX).", flush=True)
        return None, None
