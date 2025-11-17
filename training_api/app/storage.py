from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from filelock import FileLock
from joblib import dump


MODEL_DIR = Path("/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def model_paths(prefix: str = "model") -> Tuple[Path, Path]:
    ts = _timestamp()
    model_path = MODEL_DIR / f"{prefix}_{ts}.pkl"
    meta_path = MODEL_DIR / f"{prefix}_{ts}.json"
    return model_path, meta_path


def save_model(pipeline, metadata: Dict[str, Any]) -> Dict[str, str]:
    model_path, meta_path = model_paths()
    lock = FileLock(str(model_path) + ".lock")
    with lock:
        dump(pipeline, model_path)
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
    return {"model_path": str(model_path), "metadata_path": str(meta_path)}
