from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris

from feature_selection_filter.core.filter import FilterSelector
from feature_selection_frame.core.frame import FrameSelector
from feature_selection_abess.core.abess_sel import AbessSelector
from feature_selection_shap.core.shap_sel import ShapSelector
from feature_selection_permutation.core.perm_sel import PermutationSelector


def run() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    step1 = FilterSelector(k_best=3)
    X1 = step1.fit_transform(X, y)

    step2 = FrameSelector(n_features=3)
    X2 = step2.fit_transform(X1, y)

    step3 = AbessSelector()
    X3 = step3.fit_transform(X2, y)

    step4 = ShapSelector(threshold=0.01)
    X4 = step4.fit_transform(X3, y)

    step5 = PermutationSelector(tol=0.0)
    X_final = step5.fit_transform(X4, y)
    return X_final


if __name__ == "__main__":
    df_final = run()
    print(df_final.head())
