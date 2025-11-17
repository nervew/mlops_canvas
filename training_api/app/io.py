from __future__ import annotations

import io
from typing import Optional, Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile


MAX_FILE_SIZE = 50 * 1024 * 1024


def _validate_size(file: UploadFile) -> None:
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")


def load_dataset(file: Optional[UploadFile], records: Optional[list], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if file is None and records is None:
        raise HTTPException(status_code=400, detail="Provide file or records")

    if file:
        _validate_size(file)
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.DataFrame(records)

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column missing")

    y = df.pop(target)
    return df, y
