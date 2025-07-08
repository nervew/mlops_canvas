import openml
import pandas as pd

def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset Credit-G de OpenML en un DataFrame,
    eliminando la columna credit_amount para evitar problemas de unicidad.
    """
    dataset = openml.datasets.get_dataset(31)  # ID de Credit-G
    df, *_ = dataset.get_data()
    # Renombra la columna de target
    if "class" in df.columns:
        df = df.rename(columns={"class": "target"})
    # Elimina la columna problemática
    if "credit_amount" in df.columns:
        df = df.drop(columns=["credit_amount"])
    return df
