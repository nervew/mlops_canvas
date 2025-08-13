from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from onnxruntime import InferenceSession, SessionOptions
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def _ensure_str_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.map(str)
    return out

class JoblibModelExporter:
    """Exporta un pipeline sklearn ajustado a .joblib."""
    def export(self, model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

class OnnxPipelineExporter:
    """
    Exporta un *clon exportable* del pipeline sklearn:
      - Reemplaza pasos no soportados por ONNX (p.ej. IQRClipper) por etapas compatibles.
      - Mantiene el mismo ColumnTransformer y columnas.
      - Fija nombres de ENTRADA en ONNX con initial_types (usa columnas reales de X_fit).
    """
    def _clone_exportable_ct(self, preprocess, X_fit: pd.DataFrame) -> ColumnTransformer:
        """
        Crea un ColumnTransformer nuevo compatible con skl2onnx:
          - num: SimpleImputer(median) -> RobustScaler (sin IQRClipper)
          - cat: OneHotEncoder(handle_unknown='ignore', dense)
          - time: passthrough
        Usa las mismas columnas que el CT original.
        """
        assert hasattr(preprocess, "named_steps") and "ct" in preprocess.named_steps, \
            "Se esperaba un Pipeline con step 'ct' (ColumnTransformer)."
        ct_orig: ColumnTransformer = preprocess.named_steps["ct"]

        # Extraer especificaciones
        transformers = []
        for name, trans, cols in ct_orig.transformers_:
            if name == "num" and trans != "drop" and trans is not None:
                # Construir pipeline NUM exportable (sin IQRClipper)
                num_pipe = SkPipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc",  RobustScaler(with_centering=True, with_scaling=True,
                                         quantile_range=(25.0, 75.0))),
                ])
                transformers.append(("num", num_pipe, cols))
            elif name == "cat" and trans != "drop" and trans is not None:
                # OneHotEncoder denso y estable
                ohe_kwargs = dict(handle_unknown="ignore", dtype=np.float32)
                try:
                    ohe_kwargs["sparse_output"] = False
                except Exception:
                    ohe_kwargs["sparse"] = False
                cat_pipe = SkPipeline(steps=[("ohe", OneHotEncoder(**ohe_kwargs))])
                transformers.append(("cat", cat_pipe, cols))
            elif name == "time":
                # passthrough (ya coercionadas a float32 fuera)
                transformers.append(("time", "passthrough", cols))
            else:
                # copy tal cual si es drop / passthrough / None
                transformers.append((name, trans, cols))

        ct_export = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,
            n_jobs=None,
            verbose=False,
            verbose_feature_names_out=False,
        )
        # Ajustar el CT exportable con X_fit (coercido) para fijar estadísticas
        ct_export.fit(X_fit)
        return ct_export

    def export(self, preprocess, X_fit: pd.DataFrame, onnx_path: Path, verify: bool = True) -> None:
        X_fit = _ensure_str_columns_df(X_fit)

        # 1) Construir clon exportable (sin IQRClipper) y encajarlo
        ct_export = self._clone_exportable_ct(preprocess, X_fit)
        exportable = SkPipeline(steps=[("ct", ct_export)])

        # 2) initial_types con NOMBRES REALES y tipos correctos
        init_types = []
        for c in X_fit.columns:
            if pd.api.types.is_numeric_dtype(X_fit[c]):
                init_types.append((c, FloatTensorType([None, 1])))
            else:
                init_types.append((c, StringTensorType([None, 1])))

        # 3) Convertir a ONNX
        onx = convert_sklearn(
            exportable,
            initial_types=init_types,
            target_opset=19,
            options={"zipmap": False},
        )
        onx_bytes = onx.SerializeToString()

        # 4) Guardar
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with open(onnx_path, "wb") as f:
            f.write(onx_bytes)

        # 5) Verificación ligera
        if verify:
            try:
                sess_opts = SessionOptions()
                sess = InferenceSession(onnx_path.as_posix(), sess_options=sess_opts, providers=["CPUExecutionProvider"])
                Xmini = X_fit.iloc[: min(4, len(X_fit))].copy()
                feed = {}
                for inp in sess.get_inputs():
                    name = inp.name
                    col = Xmini[name]
                    t = inp.type.lower()
                    if "tensor(string)" in t:
                        arr = col.astype(object).values.reshape((-1, 1))
                    elif "tensor(float)" in t:
                        arr = col.astype(np.float32).values.reshape((-1, 1))
                    elif "tensor(double)" in t:
                        arr = col.astype(np.float64).values.reshape((-1, 1))
                    else:
                        arr = col.to_numpy().reshape((-1, 1))
                    feed[name] = arr
                sess.run(None, feed)
            except Exception as e:
                import warnings
                warnings.warn(f"[Export ONNX] Verificación falló: {e}", RuntimeWarning)
