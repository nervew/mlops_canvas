from __future__ import annotations

import re
from typing import Optional, Sequence, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder

from ..ports import ICategoricalEncoder


def _is_sklearn_ge_12() -> bool:
    """Devuelve True si scikit-learn >= 1.2 (para usar 'sparse_output')."""
    try:
        import sklearn
        ver = getattr(sklearn, "__version__", "0.0")
        major, minor, *_ = [int(x) for x in ver.split(".")]
        return (major, minor) >= (1, 2)
    except Exception:
        return False


def _sanitize_token(x: object) -> str:
    """Token seguro para nombre de columna."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "missing"
    s = str(x).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\.:]", "_", s)
    return s if s else "empty"


def _detect_categorical_columns(df: pd.DataFrame, include_datetimes: bool) -> List[str]:
    """
    Detecta columnas categóricas candidatas:
      - object, string, category, bool
      - (opcional) datetime64[ns] y datetimetz (si include_datetimes=True)
    """
    obj = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)
    dt_naive: List[str] = []
    dt_tz: List[str] = []
    if include_datetimes:
        dt_naive = list(df.select_dtypes(include=["datetime64[ns]"]).columns)
        # compatibilidad para tz-aware en distintas pandas
        dt_tz = [c for c in df.columns
                 if "DatetimeTZDtype" in str(df[c].dtype) or "datetimetz" in str(df[c].dtype)]
    return list(dict.fromkeys(obj + dt_naive + dt_tz))


def _coerce_datetime_to_str(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Convierte columnas datetime (naïve o tz-aware) a string ISO.
    Rellena NaN con 'missing'. Devuelve copia.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        dtype_s = str(out[c].dtype)
        if (
            pd.api.types.is_datetime64_any_dtype(out[c])
            or "DatetimeTZDtype" in dtype_s
            or "datetimetz" in dtype_s
            or dtype_s.startswith("datetime64[ns")
        ):
            out[c] = pd.to_datetime(out[c], errors="coerce") \
                       .dt.strftime("%Y-%m-%d %H:%M:%S") \
                       .fillna("missing") \
                       .astype("object")
    return out


class OneHotEncoderAdapter(ICategoricalEncoder, BaseEstimator, TransformerMixin):
    """
    One-Hot robusto y amigable con ONNX:
      - Selección automática de columnas categóricas (o usar 'columns').
      - (Opcional) Convierte datetime → string y rellena faltantes con 'missing'.
      - handle_unknown='ignore' por defecto (evita fallos en producción/ONNX).
      - Compatible con sklearn <1.2 ('sparse') y >=1.2 ('sparse_output').
      - Devuelve DataFrame con nombres de columnas estables.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop: Union[str, Sequence[str], None] = None,
        handle_unknown: str = "ignore",
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        dtype: Union[np.dtype, type] = np.float32,
        return_dense: bool = True,
        treat_datetime_as_categorical: bool = False,  # NUEVO: por defecto NO OHE de datetimes
    ) -> None:
        self.columns = list(columns) if columns is not None else None
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.dtype = dtype
        self.return_dense = return_dense
        self.treat_datetime_as_categorical = treat_datetime_as_categorical

        # Atributos tras fit
        self.columns_: List[str] = []
        self.encoder_: Optional[SkOneHotEncoder] = None
        self.feature_names_out_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    # ----------------- utilidades internas -----------------

    def _build_encoder(self) -> SkOneHotEncoder:
        params = dict(handle_unknown=self.handle_unknown, dtype=self.dtype)
        if self.drop is not None:
            params["drop"] = self.drop
        if self.min_frequency is not None:
            params["min_frequency"] = self.min_frequency
        if self.max_categories is not None:
            params["max_categories"] = self.max_categories

        # Compatibilidad de versión
        if _is_sklearn_ge_12():
            params["sparse_output"] = not self.return_dense
        else:
            params["sparse"] = not self.return_dense

        # Si alguna key no existe en la versión actual, la ignoramos
        try:
            enc = SkOneHotEncoder(**params)
        except TypeError:
            valid = SkOneHotEncoder().__init__.__code__.co_varnames
            clean = {k: v for k, v in params.items() if k in valid}
            enc = SkOneHotEncoder(**clean)
        return enc

    @staticmethod
    def _ensure_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    # ------------------- API sklearn -----------------------

    def fit(self, df: Union[pd.DataFrame, np.ndarray], y=None) -> "OneHotEncoderAdapter":
        X = self._ensure_df(df)

        # columnas a codificar
        if self.columns is None:
            cols = _detect_categorical_columns(X, include_datetimes=self.treat_datetime_as_categorical)
            self.columns_ = cols
        else:
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise KeyError(f"Columnas no encontradas para OHE: {missing}")
            self.columns_ = list(self.columns)

        if len(self.columns_) == 0:
            # No hay categóricas → encoder vacío para mantener interfaz
            self.encoder_ = self._build_encoder()
            self.encoder_.fit(pd.DataFrame(index=X.index))
            self.feature_names_out_ = np.array([], dtype=object)
            self.fitted_ = True
            return self

        # Si en columns_ hay datetimes y se optó por tratarlos como categóricos, convertir a string
        Xcat = _coerce_datetime_to_str(X[self.columns_], self.columns_)

        for c in self.columns_:
            if Xcat[c].isna().any():
                Xcat[c] = Xcat[c].astype("object").fillna("missing")

        self.encoder_ = self._build_encoder()
        self.encoder_.fit(Xcat)

        # nombres de salida
        try:
            self.feature_names_out_ = self.encoder_.get_feature_names_out(self.columns_)
        except Exception:
            # fallback para sklearn muy antiguo
            names = []
            for col, cats in zip(self.columns_, self.encoder_.categories_):
                for cat in cats:
                    names.append(f"{col}__{_sanitize_token(cat)}")
            self.feature_names_out_ = np.array(names, dtype=object)

        self.fitted_ = True
        return self

    def transform(self, df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not self.fitted_ or self.encoder_ is None or self.feature_names_out_ is None:
            raise RuntimeError("OneHotEncoderAdapter no está ajustado. Llama a fit() primero.")

        X = self._ensure_df(df).copy()

        if len(self.columns_) == 0:
            return X

        # Si faltan columnas vistas en fit, las creamos como NaN (que imputamos a 'missing')
        missing = [c for c in self.columns_ if c not in X.columns]
        for c in missing:
            X[c] = pd.NA

        Xcat = _coerce_datetime_to_str(X[self.columns_], self.columns_)
        for c in self.columns_:
            if Xcat[c].isna().any():
                Xcat[c] = Xcat[c].astype("object").fillna("missing")

        mat = self.encoder_.transform(Xcat)
        if self.return_dense and hasattr(mat, "toarray"):
            mat = mat.toarray()

        Xenc = pd.DataFrame(mat, index=X.index, columns=self.feature_names_out_)

        # Conservar orden original de no-categóricas
        rest_cols = [c for c in X.columns if c not in self.columns_]
        if rest_cols:
            out = pd.concat([X[rest_cols].reset_index(drop=True),
                             Xenc.reset_index(drop=True)], axis=1)
            out.index = X.index
            return out
        return Xenc

    def fit_transform(self, df: Union[pd.DataFrame, np.ndarray], y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def get_feature_names_out(self) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise RuntimeError("Aún no ajustado.")
        return self.feature_names_out_.copy()
