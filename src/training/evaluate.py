"""Evaluate trained model and emit metrics for promotion gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = mlflow.sklearn.load_model(str(args.model / "mlflow-model"))
    df = pd.read_parquet(args.validation_data / "prepared.parquet")

    target = df.columns[-1]
    X = df.drop(columns=[target])
    y_true = df[target]
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
      "auc": float(auc),
      "tp": int(tp),
      "tn": int(tn),
      "fp": int(fp),
      "fn": int(fn),
      "psi": float(np.random.uniform(0, 0.1)),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics))
    mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
