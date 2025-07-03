import os
from pathlib import Path

# raíz del proyecto = dos niveles arriba de este archivo
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
