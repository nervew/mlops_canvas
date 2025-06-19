from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_hist(series: pd.Series, path: Path, bins: int = 50, kde: bool = True) -> None:
    """Save a histogram plot to disk."""
    plt.figure()
    sns.histplot(series, bins=bins, kde=kde)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
