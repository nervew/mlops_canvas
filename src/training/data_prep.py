"""Data preparation script for Azure ML pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = []
    for file in args.raw_data.glob("*.parquet"):
        frames.append(pd.read_parquet(file))
    if not frames:
        raise FileNotFoundError("No input parquet files found")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna()

    output_path = args.output / "prepared.parquet"
    df.to_parquet(output_path, index=False)

    metadata = {"rows": len(df), "columns": df.columns.tolist()}
    (args.output / "metadata.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
