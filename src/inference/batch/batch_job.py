"""Batch inference job reading and writing partitioned parquet."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .schema_contract import InferenceInputSchema, InferenceOutputSchema
from ..online.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scoring-date", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scoring_date = datetime.fromisoformat(args.scoring_date).date()

    model = load_model()
    failures = 0
    total = 0
    outputs = []

    for file in args.input.glob("*.parquet"):
        df = pd.read_parquet(file)
        InferenceInputSchema.validate_dataframe(df, lazy=True)
        proba = model.predict_proba(df.values)
        pred_df = df.copy()
        pred_df["score"] = proba
        pred_df["label"] = (pred_df["score"] >= 0.5).astype(int)
        InferenceOutputSchema.validate_dataframe(pred_df, lazy=True)
        total += len(pred_df)
        outputs.append(pred_df)

    if total == 0:
        raise RuntimeError("No data for batch inference")

    result = pd.concat(outputs, ignore_index=True)
    fail_rate = failures / total
    if fail_rate > 0.005:
        raise RuntimeError(f"Failure rate {fail_rate} exceeds threshold")

    dest = args.output / f"date={scoring_date.isoformat()}"
    dest.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest / "predictions.parquet", index=False)


if __name__ == "__main__":
    main()
