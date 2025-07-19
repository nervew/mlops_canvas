"""Feature selection pipeline with multiple strategies."""

from feature_selection.filtering.core.filter_roughfs import FilterRoughFS
from feature_selection.frame.core.frame_selector import FrameSelector
from feature_selection.abess.core.abess_selector import AbessSelector
from feature_selection.shap_select.core.shap_selector import ShapSelector
from feature_selection.permutation.core.permutation_selector import (
    PermutationSelector,
)
from feature_selection.feature_selection_task import run_feature_selection

__all__ = [
    "FilterRoughFS",
    "FrameSelector",
    "AbessSelector",
    "ShapSelector",
    "PermutationSelector",
    "run_feature_selection",
]
