from ..infrastructure.openml_loader import load_dataset
from ..domain.dataset import Dataset

def ingest() -> Dataset:
    """Devuelve el dataset envuelto en la entidad Dataset."""
    df = load_dataset()
    return Dataset(data=df)