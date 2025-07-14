from __future__ import annotations

from sklearn.datasets import load_iris
import pandas as pd

from feature_selection.filtering.core.filter_roughfs import FilterRoughFS
from feature_selection.frame.core.frame_selector import FrameSelector
from feature_selection.abess.core.abess_selector import AbessSelector
from feature_selection.shap_select.core.shap_selector import ShapSelector
from feature_selection.permutation.core.permutation_selector import PermutationSelector


def run_pipeline() -> pd.DataFrame:
    X, y = load_iris(return_X_y=True, as_frame=True)
    selectors = [
        FilterRoughFS(),
        FrameSelector(),
        AbessSelector(),
        ShapSelector(),
        PermutationSelector(),
    ]

    for selector in selectors:
        X = selector.fit_transform(X, y)
    print(f"Selected features: {list(X.columns)}")
    return X


if __name__ == "__main__":
    run_pipeline()
