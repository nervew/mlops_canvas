"""Utilities for consistent MLflow usage."""
from __future__ import annotations

import mlflow


def log_lineage(run_name: str, commit_sha: str, params: dict[str, float], metrics: dict[str, float]) -> None:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git.commit", commit_sha)
        mlflow.set_tag("lineage.version", run.info.run_id)
