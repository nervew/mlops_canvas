from __future__ import annotations

from typing import List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


def _ensure_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    # Solo columnas numéricas reales (excluye datetime, bool tratados como int)
    # Si quieres incluir bool explícitamente, cambia include a ["number", "bool"]
    return list(df.select_dtypes(include=["number"]).columns)


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generador polinómico vectorizado y rápido.

    - Solo aplica sobre columnas numéricas (por defecto).
    - Por defecto EXCLUYE columnas con patrón de OHE ("__") para evitar explosión.
    - Controla complejidad con 'max_new_features'.
    - No usa loops de inserción: construye todo y concatena una vez.
    - Tolerante a columnas faltantes en producción (rellena con 0s para mantener forma).

    Parámetros
    ----------
    degree : int
        Grado polinómico (>= 2). Por defecto 2.
    interaction_only : bool
        Si True, solo términos de interacción (sin potencias). Por defecto True.
    include_bias : bool
        Si True, incluye columna de 1s. Por defecto False.
    columns : Optional[Sequence[str]]
        Subconjunto explícito de columnas numéricas a usar. Si None, autodetecta.
    exclude_ohe_like : bool
        Si True, excluye columnas cuyo nombre contiene "__" (típico de OHE). Por defecto True.
    include_input_features : bool
        Si True, deja también las columnas originales en la salida. Por defecto True.
    keep_degree_one_terms : bool
        Si False, descarta los términos de grado 1 que regresan de PolynomialFeatures
        (evitas duplicar columnas cuando include_input_features=True). Por defecto False.
    max_new_features : int
        Máximo de columnas NUEVAS a añadir (interacciones y potencias). Si se excede,
        intenta simplificar (primero interaction_only=True, luego degree=2). Por defecto 5000.
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = True,
        include_bias: bool = False,
        columns: Optional[Sequence[str]] = None,
        exclude_ohe_like: bool = True,
        include_input_features: bool = True,
        keep_degree_one_terms: bool = False,
        max_new_features: int = 5000,
    ) -> None:
        if degree < 2:
            raise ValueError("degree debe ser >= 2 para que tenga sentido.")
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.columns = list(columns) if columns is not None else None
        self.exclude_ohe_like = exclude_ohe_like
        self.include_input_features = include_input_features
        self.keep_degree_one_terms = keep_degree_one_terms
        self.max_new_features = int(max_new_features)

        # Atributos tras fit
        self.columns_: List[str] = []
        self.poly_: Optional[PolynomialFeatures] = None
        self.feature_names_poly_: Optional[np.ndarray] = None
        self._degree_used_: int = degree
        self._interaction_only_used_: bool = interaction_only
        self.fitted_: bool = False

    # ----------------- utilidades internas -----------------

    @staticmethod
    def _filter_ohe_like(cols: Sequence[str], enabled: bool) -> List[str]:
        if not enabled:
            return list(cols)
        # Heurística: columnas OHE suelen llevar separador "__"
        return [c for c in cols if "__" not in c]

    def _pick_columns(self, X: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise KeyError(f"Columnas no encontradas para polinomios: {missing}")
            chosen = list(self.columns)
        else:
            chosen = _numeric_columns(X)
        chosen = self._filter_ohe_like(chosen, self.exclude_ohe_like)
        return chosen

    def _fit_poly_with_cap(self, Xnum: pd.DataFrame) -> None:
        """
        Ajusta PolynomialFeatures respetando max_new_features.
        Intentos:
          1) degree=self.degree, interaction_only=self.interaction_only
          2) si excede -> interaction_only=True
          3) si excede -> degree=2 (y interaction_only según 2)
        """
        d = self.degree
        io = self.interaction_only

        for attempt in range(3):
            poly = PolynomialFeatures(
                degree=d, interaction_only=io, include_bias=self.include_bias
            )
            poly.fit(Xnum.values)  # rápido y seguro

            n_out = poly.n_output_features_  # total features que devuelve
            n_in = Xnum.shape[1]
            n_new = n_out - (1 if self.include_bias else 0) - n_in  # nuevas (sin bias ni grado-1)

            if n_new <= self.max_new_features:
                # Aprobado
                self.poly_ = poly
                self._degree_used_ = d
                self._interaction_only_used_ = io
                # Nombres con features originales
                try:
                    self.feature_names_poly_ = poly.get_feature_names_out(Xnum.columns.to_list())
                except Exception:
                    # fallback (nombres genéricos)
                    self.feature_names_poly_ = np.array([f"poly_{i}" for i in range(n_out)], dtype=object)
                return

            # Si excede, relajamos configuración
            if attempt == 0:
                io = True  # forzar solo interacciones
            elif attempt == 1:
                d = 2     # bajar a grado 2

        # Si aún excede, ajustamos con la última configuración y seguimos,
        # aunque supere el cap (evitamos romper el pipeline). Se puede
        # controlar aguas arriba con 'columns' para limitar.
        self.poly_ = PolynomialFeatures(
            degree=d, interaction_only=io, include_bias=self.include_bias
        )
        self.poly_.fit(Xnum.values)
        try:
            self.feature_names_poly_ = self.poly_.get_feature_names_out(Xnum.columns.to_list())
        except Exception:
            self.feature_names_poly_ = np.array(
                [f"poly_{i}" for i in range(self.poly_.n_output_features_)], dtype=object
            )
        self._degree_used_ = d
        self._interaction_only_used_ = io

    # ------------------- API sklearn -----------------------

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "PolynomialFeatureGenerator":
        Xdf = _ensure_df(X)
        cols = self._pick_columns(Xdf)

        if len(cols) == 0:
            # nada que hacer: seguimos passthrough
            self.columns_ = []
            self.poly_ = None
            self.feature_names_poly_ = np.array([], dtype=object)
            self.fitted_ = True
            return self

        self.columns_ = cols
        Xnum = Xdf[cols].copy()
        # Imputación simple para NaN → 0; evita NaNs en potencias
        Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        self._fit_poly_with_cap(Xnum)
        self.fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("PolynomialFeatureGenerator no está ajustado. Llama a fit() primero.")

        Xdf = _ensure_df(X).copy()

        # Si no se ajustó sobre columnas numéricas (caso vacío), passthrough
        if self.poly_ is None or len(self.columns_) == 0:
            return Xdf

        # Garantizar que existan todas las columnas vistas en fit y en el mismo orden
        miss = [c for c in self.columns_ if c not in Xdf.columns]
        if miss:
            # Añadimos columnas faltantes con 0 (consistente con fillna del fit)
            for c in miss:
                Xdf[c] = 0.0

        Xnum = Xdf[self.columns_].copy()
        Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        # Calculamos matriz polinómica una sola vez (rápido)
        M = self.poly_.transform(Xnum.values)  # ndarray

        # Construimos DataFrame de salida sólo con lo que queremos:
        # nombres provienen de feature_names_poly_ (alineados con M)
        names = list(self.feature_names_poly_)

        # Si no queremos duplicar grado 1 porque conservamos originales,
        # filtramos los de grado 1 (que coinciden exactamente con self.columns_)
        if not self.keep_degree_one_terms:
            # Índices de columnas "grado 1", sin bias
            idx_keep = []
            for j, name in enumerate(names):
                if name in self.columns_:
                    # grado 1 (y coincide con original) -> lo omitimos si vamos a mantener originales
                    continue
                idx_keep.append(j)
            M = M[:, idx_keep]
            names = [names[j] for j in idx_keep]

        # DataFrame polinómico
        df_poly = pd.DataFrame(M, index=Xdf.index, columns=names)

        # Columnas a mantener del original (resto)
        rest_cols = [c for c in Xdf.columns if (c not in self.columns_) or self.include_input_features]

        # Concatenación única, evita fragmentación
        out = pd.concat([Xdf[rest_cols].reset_index(drop=True),
                         df_poly.reset_index(drop=True)], axis=1)
        out.index = Xdf.index
        return out

    # ------------------- utilidades -----------------------

    def get_feature_names_out(self) -> np.ndarray:
        if self.feature_names_poly_ is None:
            raise RuntimeError("Aún no ajustado.")
        return self.feature_names_poly_.copy()

    # Para depuración/config
    def get_params_config(self) -> dict:
        return {
            "degree": self.degree,
            "interaction_only": self.interaction_only,
            "include_bias": self.include_bias,
            "columns": self.columns,
            "exclude_ohe_like": self.exclude_ohe_like,
            "include_input_features": self.include_input_features,
            "keep_degree_one_terms": self.keep_degree_one_terms,
            "max_new_features": self.max_new_features,
            "degree_used": getattr(self, "_degree_used_", None),
            "interaction_only_used": getattr(self, "_interaction_only_used_", None),
            "n_columns_in": len(self.columns_),
            "n_features_poly_out": None if self.poly_ is None else self.poly_.n_output_features_,
        }
