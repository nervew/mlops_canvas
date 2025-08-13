from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Pasos / puertos
from .step01_import import ParquetPartitionLoader
from .step03_outliers import IQRClipper
from .step07_metrics import JsonMetricsExporter
from .step08_drift import PSIDriftDetector
from .step09_export import (
    ParquetExportAdapter,
    JoblibModelExporter,
    OnnxPipelineExporter,
)

# =========================
# Utilidades
# =========================

def _is_sklearn_ge_12() -> bool:
    try:
        import sklearn
        ver = getattr(sklearn, "__version__", "0.0")
        major, minor, *_ = [int(x) for x in ver.split(".")]
        return (major, minor) >= (1, 2)
    except Exception:
        return False

def _detect_temporal_columns_raw(X: pd.DataFrame,
                                 min_valid_ratio: float = 0.8,
                                 min_unique: int = 3) -> list[str]:
    temporal: list[str] = []
    for c in X.columns:
        s = X[c]
        if getattr(s.dtype, "kind", None) == "M":
            temporal.append(c); continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= min_valid_ratio and parsed.nunique(dropna=True) >= min_unique:
                    temporal.append(c)
            except Exception:
                pass
    return list(dict.fromkeys(temporal))

def _coerce_for_pipeline(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    time_cols = _detect_temporal_columns_raw(Xc)
    for c in time_cols:
        s = pd.to_datetime(Xc[c], errors="coerce")
        days = (s - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        Xc[c] = np.nan_to_num(days.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for c in Xc.select_dtypes(include=["object", "string"]).columns.tolist():
        if not pd.api.types.is_object_dtype(Xc[c]):
            Xc[c] = Xc[c].astype(object)
    bool_cols = Xc.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        Xc[bool_cols] = Xc[bool_cols].astype(np.float32)
    num_cols = Xc.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        Xc[num_cols] = Xc[num_cols].astype(np.float32)
    return Xc

def _build_ohe_kwargs() -> dict:
    kwargs = dict(handle_unknown="ignore", dtype=np.float32)
    if _is_sklearn_ge_12():
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return kwargs

def _build_ohe_instance() -> OneHotEncoder:
    kwargs = _build_ohe_kwargs()
    try:
        return OneHotEncoder(feature_name_combiner=None, **kwargs)
    except TypeError:
        return OneHotEncoder(**kwargs)

def build_unified_preprocess(X_sample: pd.DataFrame) -> tuple[SkPipeline, dict]:
    time_cols = _detect_temporal_columns_raw(X_sample)
    Xs = _coerce_for_pipeline(X_sample)

    num_all = Xs.select_dtypes(include=["number"]).columns.tolist()
    cat_all = Xs.select_dtypes(include=["object"]).columns.tolist()

    num_cols = [c for c in num_all if c not in time_cols]
    cat_cols = [c for c in cat_all if c not in time_cols]

    num_pipe = SkPipeline(steps=[
        ("imp",  SimpleImputer(strategy="median")),
        ("iqr",  IQRClipper(factor=1.5)),
        ("sc",   RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))),
    ])
    cat_pipe = SkPipeline(steps=[("ohe", _build_ohe_instance())]) if cat_cols else "drop"
    time_pipe = "passthrough" if len(time_cols) > 0 else "drop"

    transformers = []
    if num_cols:
        transformers.append(("num",  num_pipe,  num_cols))
    if cat_cols:
        transformers.append(("cat",  cat_pipe,  cat_cols))
    if time_cols:
        transformers.append(("time", time_pipe, time_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        verbose=False,
        verbose_feature_names_out=False,
    )
    preprocess = SkPipeline(steps=[("ct", ct)])

    ordered_in_cols = [c for c in Xs.columns if c in (num_cols + cat_cols + time_cols)]
    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "time_cols": time_cols,
        "all_cols": num_cols + cat_cols + time_cols,
        "ordered_in_cols": ordered_in_cols,
    }
    return preprocess, meta

# =========================
# Orquestador
# =========================

class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        self.loader   = ParquetPartitionLoader()
        self.metrics  = JsonMetricsExporter()
        self.drift    = PSIDriftDetector()
        self.parquet_exporter = ParquetExportAdapter()
        self.joblib_exporter = JoblibModelExporter()
        self.onnx_exporter   = OnnxPipelineExporter()

        self.base_dir = Path(__file__).resolve().parents[3]
        self.preprocess: SkPipeline | None = None
        self.meta: dict | None = None

    def _export_engineering_logs(self, meta: dict, feature_names_out: list[str], iqr_bounds: dict) -> None:
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1) columnas por tipo
        with open(log_dir / "log_engineering.json", "w", encoding="utf-8") as fh:
            json.dump({
                "time_cols": meta.get("time_cols", []),
                "num_cols":  meta.get("num_cols", []),
                "cat_cols":  meta.get("cat_cols", []),
            }, fh, indent=4, ensure_ascii=False)
        print(f"[Log] Ingeniería de features exportada en {log_dir / 'log_engineering.json'}")

        # 2) nombres de salida
        with open(log_dir / "feature_names_out.json", "w", encoding="utf-8") as fh:
            json.dump(feature_names_out, fh, indent=4, ensure_ascii=False)
        print(f"[Log] Nombres de salida exportados en {log_dir / 'feature_names_out.json'}")

        # 3) límites IQR por columna numérica (para clipping fuera del ONNX)
        with open(log_dir / "iqr_bounds.json", "w", encoding="utf-8") as fh:
            json.dump(iqr_bounds, fh, indent=4, ensure_ascii=False)
        print(f"[Log] Límites IQR exportados en {log_dir / 'iqr_bounds.json'}")

    def run(self) -> None:
        # 1) Cargar particiones
        X_train, X_test, X_back, y_train, y_test, y_back = self.loader.load()

        # 2) Pipeline
        self.preprocess, self.meta = build_unified_preprocess(X_train)

        # 3) Coerción + fit/transform
        X_train_c = _coerce_for_pipeline(X_train)
        X_test_c  = _coerce_for_pipeline(X_test)
        X_back_c  = _coerce_for_pipeline(X_back)

        cols_in = (self.meta or {}).get("ordered_in_cols", [])
        if cols_in and all(c in X_train_c.columns for c in cols_in):
            X_train_c = X_train_c[cols_in]
            X_test_c  = X_test_c[cols_in]
            X_back_c  = X_back_c[cols_in]

        self.preprocess.fit(X_train_c, y_train)

        # 3.1 nombres de salida
        ct = self.preprocess.named_steps["ct"]
        try:
            feature_names_out = list(map(str, ct.get_feature_names_out(cols_in if cols_in else None)))
        except Exception as e:
            warnings.warn(f"No se pudieron obtener feature_names_out: {e}", RuntimeWarning)
            feature_names_out = [str(i) for i in range(self.preprocess.transform(X_train_c).shape[1])]

        # 3.2 extraer límites IQR del step numérico para exportarlos
        iqr_bounds: dict[str, list[float]] = {}
        try:
            num_pipe: SkPipeline = ct.named_transformers_.get("num", None)
            if isinstance(num_pipe, SkPipeline) and "iqr" in num_pipe.named_steps:
                bounds = num_pipe.named_steps["iqr"].bounds_
                # JSON-serializable
                iqr_bounds = {k: [float(v[0]), float(v[1])] for k, v in bounds.items()}
        except Exception as e:
            warnings.warn(f"No se pudieron extraer límites IQR: {e}", RuntimeWarning)

        # 3.3 Transform + DF con nombres
        def to_df_named(arr, index):
            df = pd.DataFrame(arr, index=index, columns=feature_names_out)
            df.columns = df.columns.map(str)
            return df

        X_train_t = to_df_named(self.preprocess.transform(X_train_c), X_train.index)
        X_test_t  = to_df_named(self.preprocess.transform(X_test_c),  X_test.index)
        X_back_t  = to_df_named(self.preprocess.transform(X_back_c),  X_back.index)

        # 4) Exportaciones
        transformers_dir = self.base_dir / "transformers"
        self.joblib_exporter.export(self.preprocess, transformers_dir / "transformador_inicial.joblib")
        # ONNX: usa clon exportable (sin IQR) internamente
        self.onnx_exporter.export(self.preprocess, X_train_c, transformers_dir / "transformador_inicial.onnx", verify=True)

        # 4.1) Logs
        self._export_engineering_logs(self.meta or {}, feature_names_out, iqr_bounds)

        # 5) Exportar datasets procesados con nombres
        X_train_final = X_train_t.copy(); X_train_final["target"] = y_train.values
        X_test_final  = X_test_t.copy();  X_test_final["target"]  = y_test.values
        X_back_final  = X_back_t.copy();  X_back_final["target"]  = y_back.values
        ParquetExportAdapter().export(X_train_final, X_test_final, X_back_final)

        # 6) Métricas y drift
        report_dir = self.base_dir / "output" / "report_engineering"
        report_dir.mkdir(parents=True, exist_ok=True)
        JsonMetricsExporter().export(X_train_final, str(report_dir / "feature_metrics.json"))

        drift = PSIDriftDetector().compute(X_train_t, X_test_t)
        with open(report_dir / "drift.json", "w", encoding="utf-8") as fh:
            json.dump(drift, fh, indent=4)

def run_pipeline() -> None:
    FeatureEngineeringPipeline().run()

if __name__ == "__main__":
    run_pipeline()
