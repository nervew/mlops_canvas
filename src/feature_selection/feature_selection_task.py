from __future__ import annotations

from typing import Dict, List
from time import perf_counter

import numpy as np
import pandas as pd

from feature_selection.filtering.core.filter_roughfs import FilterRoughFS
from feature_selection.frame.core.frame_selector import FrameSelector
from feature_selection.abess.core.abess_selector import AbessSelector
from feature_selection.shap_select.core.shap_selector import ShapSelector
from feature_selection.permutation.core.permutation_selector import PermutationSelector


TECHNIQUE_MAP = {
    "filter": FilterRoughFS,
    "roughfs": FilterRoughFS,
    "frame": FrameSelector,
    "abess": AbessSelector,
    "shap": ShapSelector,
    "permutation": PermutationSelector,
}


def _validate_inputs(
    dataframe: pd.DataFrame, target_column: str, sample_fraction: float
) -> None:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not in dataframe")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")


def _prepare_data(
    dataframe: pd.DataFrame, target_column: str, sample_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_full = dataframe.drop(columns=[target_column])
    y_full = dataframe[target_column]
    if sample_fraction < 1.0:
        df_sample = dataframe.sample(frac=sample_fraction, random_state=seed)
    else:
        df_sample = dataframe
    X_sample = df_sample.drop(columns=[target_column])
    y_sample = df_sample[target_column]
    return X_full, X_sample, y_full, y_sample


def run_feature_selection(
    dataframe: pd.DataFrame,
    target_column: str,
    technique: str = "filter",
    sample_fraction: float = 1.0,
    random_state: int = 42,
    batch_size: int = 1_000_000,
    save_logs: bool = False,
    output_selected_only: bool = True,
) -> Dict[str, object]:
    """Run scalable feature selection on a pandas DataFrame.

    Args:
        dataframe: Input tabular data.
        target_column: Name of the supervised target column.
        technique: Selection technique to use or ``"all"`` for the full pipeline.
        sample_fraction: Fraction of data for expensive techniques.
        random_state: Random seed for reproducibility.
        batch_size: Currently unused placeholder for future batching.
        save_logs: Whether to return timing information.
        output_selected_only: If ``True`` return only selected columns.

    Returns:
        Dictionary with ``selected_columns`` and optionally ``dropped_columns``
        and ``selection_report``.

    Raises:
        ValueError: If inputs are invalid or technique is unsupported.

    Examples:
        >>> import pandas as pd
        >>> from feature_selection.feature_selection_task import run_feature_selection
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
        >>> res = run_feature_selection(df, "target", technique="filter")
        >>> res["selected_columns"]
        ['a', 'b']
    """
    _validate_inputs(dataframe, target_column, sample_fraction)

    np.random.seed(random_state)

    technique = technique.lower()
    if technique != "all" and technique not in TECHNIQUE_MAP:
        raise ValueError(f"Unsupported technique: {technique}")

    X_full, X_sample, y_full, y_sample = _prepare_data(
        dataframe, target_column, sample_fraction, random_state
    )

    steps: List = []
    if technique == "all":
        steps = [
            FilterRoughFS(),
            FrameSelector(),
            AbessSelector(),
            ShapSelector(),
            PermutationSelector(),
        ]
    else:
        steps = [TECHNIQUE_MAP[technique]()]  # type: ignore

    logs: List[Dict[str, float]] = []
    for step in steps:
        start = perf_counter()
        step.fit(X_sample, y_sample)
        X_sample = step.transform(X_sample)
        X_full = step.transform(X_full)
        duration = perf_counter() - start
        if save_logs:
            logs.append(
                {
                    "step": step.__class__.__name__,
                    "time_sec": round(duration, 4),
                    "n_features": len(X_full.columns),
                }
            )

    selected_cols = X_full.columns.tolist()
    dropped_cols = [
        c for c in dataframe.columns if c not in selected_cols and c != target_column
    ]

    result: Dict[str, object] = {"selected_columns": selected_cols}
    if not output_selected_only:
        result["dropped_columns"] = dropped_cols
    if save_logs:
        result["selection_report"] = {"logs": logs}
    return result
