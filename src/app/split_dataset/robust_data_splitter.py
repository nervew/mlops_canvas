"""Herramientas para dividir datasets en subconjuntos robustos.

El módulo ofrece utilidades para crear splits train, test y backtest con
estratificación, métricas de estabilidad y reportes visuales.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import entropy, ks_2samp
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union


class RobustDataSplitter:
    """Divide un DataFrame en train, test y backtest de forma estable."""

    def __init__(
        self,
        df: pd.DataFrame,
        split_method: str = 'random',
        target_column: Optional[str] = None,
        stratify_columns: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        metric_column: Optional[str] = None,
        train_size: Union[float, str, pd.Timestamp] = 0.6,
        test_size: Union[float, str, pd.Timestamp] = 0.2,
        backtest_size: Union[float, str, pd.Timestamp] = 0.2,
        memory_threshold: float = 500e6,
        chunk_size: int = 1_000_000
    ) -> None:
        """Inicializa el splitter con parámetros de configuración básicos."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df debe ser un pandas DataFrame.')
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

    def _validate_parameters(self) -> None:
        """Valida parámetros según el método de división."""
        métodos = ['random', 'time', 'metric']
        if self.split_method not in métodos:
            raise ValueError(f"split_method debe ser uno de {métodos}.")
        if self.split_method == 'random':
            total = (
                self.train_size + self.test_size + self.backtest_size
            )  # type: ignore[operator]
            if not np.isclose(total, 1.0):
                raise ValueError(
                    'train_size + test_size + backtest_size debe sumar 1.0.'
                )
            if not self.target_column:
                raise ValueError(
                    'Debe especificar target_column para split aleatorio.'
                )
        if self.split_method == 'time' and not self.time_column:
            raise ValueError(
                'Debe especificar time_column para split por tiempo.'
            )
        if self.split_method == 'metric':
            if not self.metric_column:
                raise ValueError(
                    'Debe especificar metric_column para split por métrica.'
                )
            total = (
                self.train_size + self.test_size + self.backtest_size
            )  # type: ignore[operator]
            if not np.isclose(total, 1.0):
                raise ValueError(
                    'train_size + test_size + backtest_size debe sumar 1.0.'
                )

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ejecuta el split según el método configurado."""
        if self._is_large_dataset():
            return self._memory_optimized_split()
        if self.split_method == 'random':
            return self._random_split()
        if self.split_method == 'time':
            return self._time_split()
        return self._metric_split()

    def calculate_metrics(
        self,
        ref_df: Optional[pd.DataFrame] = None,
        curr_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        buckets: int = 10
    ) -> pd.DataFrame:
        """Calcula métricas PSI, JS, KS y diferencias de medias/varianzas."""
        ref = ref_df or self.train_df  # type: ignore
        curr = curr_df or self.test_df  # type: ignore
        if columns is None:
            cols = [
                col for col in [self.target_column, *self.stratify_columns]
                if col is not None
            ]
        else:
            cols = columns
        report = []
        for col in cols:
            exp = ref[col].dropna()  # type: ignore
            act = curr[col].dropna()  # type: ignore
            psi_val = self._psi(exp, act, buckets)
            js_val = self._jensen_shannon(exp, act, buckets)
            ks_stat, ks_p = self._ks_test(exp, act)
            mean_diff, var_diff = self._mean_var_diff(exp, act)
            report.append({
                'column': col,
                'psi': psi_val,
                'js_divergence': js_val,
                'ks_stat': ks_stat,
                'ks_pvalue': ks_p,
                'mean_diff': mean_diff,
                'var_diff': var_diff
            })
        df_report = pd.DataFrame(report).set_index('column')
        return df_report

    def generate_visual_report(
        self,
        columns: Optional[List[str]] = None
    ) -> None:
        """Genera reportes visuales comparativos por columna."""
        seen = set()
        cols = []
        metric_cols: List[Optional[str]] = []
        if self.metric_column:
            metric_cols.append(self.metric_column)
        default_cols: List[Optional[str]] = [
            self.target_column,
            *self.stratify_columns,
        ]
        source_cols = columns or [
            col for col in default_cols + metric_cols if col is not None
        ]
        for col in source_cols:
            if col not in seen:
                seen.add(col)
                cols.append(col)

        for col in cols:
            plt.figure(figsize=(10, 6))
            data = [
                self.train_df[col].dropna(),  # type: ignore[index]
                self.test_df[col].dropna(),
                self.backtest_df[col].dropna()
            ]
            labels = ['train', 'test', 'backtest']
            # Histograma
            plt.subplot(3, 1, 1)
            for series, label in zip(data, labels):
                plt.hist(series, bins=20, alpha=0.5, label=label)
            plt.title(f'Histogram of {col}')
            plt.legend()
            # Boxplot
            plt.subplot(3, 1, 2)
            plt.boxplot(data, labels=labels)
            plt.title(f'Boxplot of {col}')
            # CDF
            plt.subplot(3, 1, 3)
            for series, label in zip(data, labels):
                sorted_series = np.sort(series)
                cdf = np.arange(len(sorted_series)) / float(len(sorted_series))
                plt.plot(sorted_series, cdf, label=label)
            plt.title(f'Cumulative Distribution of {col}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _random_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Realiza un split aleatorio estratificado en tres subconjuntos."""
        strata = self.df[self.target_column].astype(str)  # type: ignore
        for col in self.stratify_columns:
            strata += '_' + self.df[col].astype(str)
        temp_df, back = train_test_split(
            self.df,
            test_size=self.backtest_size,  # type: ignore[arg-type]
            stratify=strata,
            random_state=42
        )
        rel_test = self.test_size / (
            self.train_size + self.test_size
        )  # type: ignore[operator]
        strata_temp = strata.loc[temp_df.index]
        train, test = train_test_split(
            temp_df,
            test_size=rel_test,
            stratify=strata_temp,
            random_state=42
        )
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    def _time_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide cronológicamente según la columna temporal configurada."""
        df_sorted = self.df.sort_values(by=self.time_column)  # type: ignore
        if isinstance(self.train_size, float):
            n = len(df_sorted)
            i1 = int(self.train_size * n)  # type: ignore
            i2 = int((self.train_size + self.test_size) * n)  # type: ignore
            train = df_sorted.iloc[:i1]
            test = df_sorted.iloc[i1:i2]
            back = df_sorted.iloc[i2:]
        else:
            cut1 = pd.to_datetime(self.train_size)
            cut2 = pd.to_datetime(self.test_size)
            time_column = self.df[self.time_column]
            train_mask = time_column <= cut1
            between_cut = (time_column > cut1) & (time_column <= cut2)
            train = self.df[train_mask]  # type: ignore[index]
            test = self.df[between_cut]
            back = self.df[time_column > cut2]
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    def _metric_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Genera splits basados en cuantiles de una métrica numérica."""
        series = self.df[self.metric_column]  # type: ignore
        q1 = series.quantile(self.train_size)  # type: ignore
        q2 = series.quantile(self.train_size + self.test_size)  # type: ignore
        train = self.df[series <= q1]
        between_q = (series > q1) & (series <= q2)
        test = self.df[between_q]
        back = self.df[series > q2]
        self.train_df, self.test_df, self.backtest_df = train, test, back
        return train, test, back

    def _is_large_dataset(self) -> bool:
        """Indica si el dataset supera el umbral de memoria configurado."""
        total_mem = self.df.memory_usage(deep=True).sum()
        return total_mem > self.memory_threshold

    def _memory_optimized_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ejecuta splits por chunks para datasets grandes."""
        chunks_tr, chunks_te, chunks_ba = [], [], []
        n = len(self.df)
        for start in range(0, n, self.chunk_size):
            stop = start + self.chunk_size
            chunk = self.df.iloc[start:stop]
            splitter = RobustDataSplitter(
                chunk,
                split_method=self.split_method,
                target_column=self.target_column,
                stratify_columns=self.stratify_columns,
                time_column=self.time_column,
                metric_column=self.metric_column,
                train_size=self.train_size,  # type: ignore[arg-type]
                test_size=self.test_size,  # type: ignore[arg-type]
                backtest_size=self.backtest_size,  # type: ignore[arg-type]
                memory_threshold=float('inf'),
                chunk_size=self.chunk_size
            )
            tr, te, ba = splitter.split_data()
            chunks_tr.append(tr)
            chunks_te.append(te)
            chunks_ba.append(ba)
        self.train_df = pd.concat(chunks_tr)
        self.test_df = pd.concat(chunks_te)
        self.backtest_df = pd.concat(chunks_ba)
        return self.train_df, self.test_df, self.backtest_df

    def _psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        buckets: int = 10
    ) -> float:
        """Calcula el Population Stability Index entre dos distribuciones."""
        percentiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        exp_perc = np.histogram(expected, bins=percentiles)[0] / len(expected)
        act_perc = np.histogram(actual, bins=percentiles)[0] / len(actual)
        eps = 1e-8
        exp_perc = np.where(exp_perc == 0, eps, exp_perc)
        act_perc = np.where(act_perc == 0, eps, act_perc)
        ratio = np.divide(exp_perc, act_perc)
        diff = exp_perc - act_perc
        return float(np.sum(diff * np.log(ratio)))

    def _jensen_shannon(
        self,
        expected: pd.Series,
        actual: pd.Series,
        buckets: int = 10
    ) -> float:
        """Calcula la divergencia de Jensen-Shannon."""
        cuts = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        p = np.histogram(expected, bins=cuts)[0] / len(expected)
        q = np.histogram(actual, bins=cuts)[0] / len(actual)
        m = 0.5 * (p + q)
        js_value = 0.5 * (entropy(p, m) + entropy(q, m))
        return float(js_value)

    def _ks_test(
        self,
        expected: pd.Series,
        actual: pd.Series
    ) -> Tuple[float, float]:
        """Ejecuta una prueba KS entre las series entregadas."""
        stat, pvalue = ks_2samp(expected, actual)
        return float(stat), float(pvalue)

    def _mean_var_diff(
        self,
        expected: pd.Series,
        actual: pd.Series
    ) -> Tuple[float, float]:
        """Obtiene diferencias de medias y varianzas entre dos series."""
        mean_delta = float(expected.mean() - actual.mean())
        var_delta = float(expected.var() - actual.var())
        return mean_delta, var_delta

# Ejemplo de uso:
# if __name__ == '__main__':
#     df = pd.read_csv('data.csv', parse_dates=['date_col'])
#     splitter = RobustDataSplitter(
#         df,
#         split_method='random',
#         target_column='y',
#         stratify_columns=['category'],
#         train_size=0.7,
#         test_size=0.2,
#         backtest_size=0.1
#     )
#     train_df, test_df, backtest_df = splitter.split_data()
#     metrics = splitter.calculate_metrics()
#     print(metrics)
#     splitter.generate_visual_report()
