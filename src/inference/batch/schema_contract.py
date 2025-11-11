"""Schema contracts for batch inference."""
from __future__ import annotations

import pandera as pa


class InferenceInputSchema(pa.SchemaModel):
    feature_1: pa.Float
    feature_2: pa.Float


class InferenceOutputSchema(InferenceInputSchema):
    score: pa.Float
    label: pa.Int
