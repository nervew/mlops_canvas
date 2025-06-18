from __future__ import annotations

import numpy as np
import pandas as pd


def generate_data(
    n_samples: int = 500, task_type: str = "classification"
) -> pd.DataFrame:
    """Generate synthetic dataset supporting multiple task types."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n_samples, freq="D"),
            "category": rng.choice(["A", "B", "C"], size=n_samples),
            "num1": rng.normal(size=n_samples),
            "num2": rng.uniform(0, 100, size=n_samples),
        }
    )

    if task_type == "classification":
        df["target"] = rng.integers(0, 2, size=n_samples)
    elif task_type in {"regression", "forecast"}:
        noise = rng.normal(scale=0.5, size=n_samples)
        df["target"] = df["num1"] * 0.3 + df["num2"] * 0.1 + noise
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return df
