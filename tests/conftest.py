import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame with numeric and categorical columns."""
    return pd.DataFrame({
        "num": [1, 2, 3, 4],
        "cat": ["a", "b", "a", "b"],
    })
