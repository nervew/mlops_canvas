"""
robust_data_splitter.py

Clase RobustDataSplitter para dividir datasets en train, test y backtest
garantizando balanceo y estabilidad. Incluye:

- Splits aleatorio, temporal y basado en cuantiles de una métrica
- Cálculo de PSI, KS, Gini, divergencia JS, etc.
- Limpieza robusta de columnas numéricas para evitar errores con strings
- Modo optimizado para datasets grandes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt


class RobustDataSplitter:
    # --------------------------- CONSTRUCTOR --------------------------- #
    def __init__(
        self,
        df: pd.DataFrame,
        split_method: str = "random",
        target_column: Optional[str] = None,
        stratify_columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        metric_column: Optional[str] = None,
        train_size: Union[float, str, pd.Timestamp] = 0.6,
        test_size: Union[float, str, pd.Timestamp] = 0.2,
        backtest_size: Union[float, str, pd.Timestamp] = 0.2,
        memory_threshold: float = 500e6,
        chunk_size: int = 1_000_000,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df debe ser un pandas DataFrame.")
        self.df = df.copy()
        self.split_method = split_method
        self.target_column = target_column
        self.stratify_columns = stratify_columns or []
        self.time_column = time_column
        self.metric_column = metric_column
        self.train_size = train_size
        self.test_size = test_size
        self.backtest_size = backtest_size
        self.memory_threshold = memory_threshold
        self.chunk_size = chunk_size

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.backtest_df: Optional[pd.DataFrame] = None
        self._validate_parameters()

    # ----------------------- VALIDACIÓN PARÁMETROS --------------------- #
    def _validate_parameters(self) -> None:
        métodos = ["random", "time", "metric"]
        if self.split_method not in métodos:
            raise ValueError(f"split_method debe ser uno de {métodos}.")

        if self.split_method == "random":
            total = self.train_size + self.test_size + self.backtest_size  # type: ignore
            if not np.isclose(total, 1.0):
                raise ValueError("train_size + test_size + backtest_size debe sumar 1.0.")
            if not self.target_column:
                raise ValueError("Debe especificar target_column para split aleatorio.")

        if self.split_method == "time" and not self.time_column:
            raise ValueError("Debe especificar time_column para split por tiempo.")

        if self.split_method == "metric":
            if not self.metric_column:
                raise ValueError("Debe especificar metric_column para split por métrica.")
            total = self.train_size + self.test_size + self.backtest_size  # type: ignore
            if not np.isclose(total, 1.0):
                raise ValueError("train_size + test_size + backtest_size debe sumar 1.0.")

    # --------------------------- SPLIT DATOS --------------------------- #
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._is_large_dataset():
            return self._memory_optimized_split()
        if self.split_method == "random":
            return self._random_split()
        if self.split_method == "time":
            return self._time_split()
        return self._metric_split()

    # -------------------- CÁLCULO DE MÉTRICAS -------------------------- #
    @staticmethod
    def _to_numeric(series: pd.Series) -> np.ndarray:
        """
        Limpia la serie para convertirla a float:
        - Elimina símbolos no numéricos ($, %, comas, espacios…)
        - Fuerza conversión; los no convertibles se vuelven NaN y luego se descartan
        """
        serie = (
            series.astype(str)
            .str.replace(r"[^\d\.\-eE]", "", regex=True)  # quita cualquier cosa que no sea dígito, punto o signo
            .replace("", np.nan)
        )
        return pd.to_numeric(serie, errors="coerce").dropna().values

    def calculate_metrics(
        self,
        ref_df: Optional[pd.DataFrame] = None,
        curr_df: Optional[pd.DataFrame] = None,
        buckets: int = 10,
    ) -> pd.DataFrame:
        """
        Calcula PSI, KS y Gini para TODAS las columnas numéricas de train vs test.

        Parameters
        ----------
        ref_df : DataFrame de referencia (default = train)
        curr_df : DataFrame a comparar (default = test)
        buckets : número de bins para PSI
        """
        ref = ref_df or self.train_df  # type: ignore
        curr = curr_df or self.test_df  # type: ignore

        # Seleccionamos solo columnas numéricas
        numeric_cols = ref.select_dtypes(include=["number"]).columns
        # Si el target es numérico pero NO quieres medirlo, exclúyelo aquí
        # numeric_cols = [c for c in numeric_cols if c != self.target_column]

        report = []
        for col in numeric_cols:
            exp = self._to_numeric(ref[col])
            act = self._to_numeric(curr[col])

            if len(exp) == 0 or len(act) == 0:
                continue  # evita divisiones por cero

            psi_val = self._psi(exp, act, buckets)
            ks_val, _ = self._ks_test(exp, act)
            gini_val = 2 * ks_val - 1  # aproximación de Gini a partir de KS
            report.append(
                {"psi": psi_val, "ks": ks_val, "gini": gini_val, "column": col}
            )

        return pd.DataFrame(report).set_index("column")

    # ----------------------- MÉTODOS AUXILIARES MÉTRICAS --------------- #
    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        percentiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        exp_perc = np.histogram(expected, bins=percentiles)[0] / len(expected)
        act_perc = np.histogram(actual, bins=percentiles)[0] / len(actual)
        eps = 1e-8
        exp_perc = np.where(exp_perc == 0, eps, exp_perc)
        act_perc = np.where(act_perc == 0, eps, act_perc)
        return float(np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc)))

    @staticmethod
    def _ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        stat, pvalue = ks_2samp(expected, actual)
        return float(stat), float(pvalue)

    # -------------------- MÉTODOS DE SPLIT ESPECÍFICOS ------------------ #
    def _random_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        strata = self.df[self.target_column].astype(str)  # type: ignore
        for col in self.stratify_columns:
            strata += "_" + self.df[col].astype(str)
        temp_df, back = train_test_split(
            self.df,
            test_size=self.backtest_size,  # type: ignore
            stratify=strata,
            random_state=42,
        )
        rel_test = self.test_size / (self.train_size + self.test_size)  # type: ignore
        strata_temp = strata.loc[temp_df.index]
        train, test = train_test_split(
            temp_df, test_size=rel_test, stratify=strata_temp, random_state=42
        )
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    def _time_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = self.df.sort_values(by=self.time_column)  # type: ignore
        n = len(df_sorted)
        i1 = int(self.train_size * n)  # type: ignore
        i2 = int((self.train_size + self.test_size) * n)  # type: ignore
        train = df_sorted.iloc[:i1]
        test = df_sorted.iloc[i1:i2]
        back = df_sorted.iloc[i2:]
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    def _metric_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        series = self.df[self.metric_column]  # type: ignore
        q1 = series.quantile(self.train_size)  # type: ignore
        q2 = series.quantile(self.train_size + self.test_size)  # type: ignore
        train = self.df[series <= q1]
        test = self.df[(series > q1) & (series <= q2)]
        back = self.df[series > q2]
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    # ----------------- UTILIDADES PARA DATASETS GRANDES ---------------- #
    def _is_large_dataset(self) -> bool:
        return self.df.memory_usage(deep=True).sum() > self.memory_threshold

    def _memory_optimized_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        chunks_tr, chunks_te, chunks_ba = [], [], []
        n = len(self.df)
        for start in range(0, n, self.chunk_size):
            chunk = self.df.iloc[start : start + self.chunk_size]
            splitter = RobustDataSplitter(
                chunk,
                split_method=self.split_method,
                target_column=self.target_column,
                stratify_columns=self.stratify_columns,
                time_column=self.time_column,
                metric_column=self.metric_column,
                train_size=self.train_size,  # type: ignore
                test_size=self.test_size,  # type: ignore
                backtest_size=self.backtest_size,  # type: ignore
                memory_threshold=float("inf"),
                chunk_size=self.chunk_size,
            )
            tr, te, ba = splitter.split_data()
            chunks_tr.append(tr)
            chunks_te.append(te)
            chunks_ba.append(ba)
        self.train_df = pd.concat(chunks_tr)
        self.test_df = pd.concat(chunks_te)
        self.backtest_df = pd.concat(chunks_ba)
        return self.train_df, self.test_df, self.backtest_df

    # ------------------------------------------------------------------ #
    # ------------------ VISUALIZACIÓN RESUMIDA (opcional) --------------#
    # ------------------------------------------------------------------ #
    def generate_visual_report(self, columns: Optional[List[str]] = None) -> None:
        cols = columns or list(
            {
                *(self.stratify_columns),
                self.target_column,
                self.metric_column,
            }
        )
        for col in cols:
            plt.figure(figsize=(10, 6))
            data = [
                self.train_df[col].dropna(),  # type: ignore
                self.test_df[col].dropna(),  # type: ignore
                self.backtest_df[col].dropna(),  # type: ignore
            ]
            labels = ["train", "test", "backtest"]

            plt.subplot(3, 1, 1)
            for d, label in zip(data, labels):
                plt.hist(d, bins=20, alpha=0.5, label=label)
            plt.title(f"Histograma de {col}")
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.boxplot(data, labels=labels)
            plt.title(f"Boxplot de {col}")

            plt.subplot(3, 1, 3)
            for d, label in zip(data, labels):
                sorted_d = np.sort(d)
                cdf = np.arange(len(sorted_d)) / float(len(sorted_d))
                plt.plot(sorted_d, cdf, label=label)
            plt.title(f"CDF acumulada de {col}")
            plt.legend()
            plt.tight_layout()
            plt.show()
