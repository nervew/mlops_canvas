from sklearn.datasets import load_iris
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Carga el dataset de iris en un DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = iris.target
    return df
