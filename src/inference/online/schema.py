"""Pydantic models for online inference."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: List[float] = Field(min_length=1)


class PredictionResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label: int = Field(ge=0, le=1)
