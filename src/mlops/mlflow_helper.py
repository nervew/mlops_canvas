from __future__ import annotations

"""Utility class to centralise MLflow interactions on Databricks.

This module defines :class:`MLflowHelper`, a thin wrapper around the
:mod:`mlflow` APIs providing convenient configuration, automatic run
management and retry logic for transient errors.  The class is designed to
run on Databricks but remains compatible with local backends.  All methods are
thread safe and avoid leaking credentials.
"""

from contextlib import contextmanager
import logging
import os
import time
from threading import Lock
from typing import Any, Dict, Iterable, Optional, Sequence

import mlflow


LOGGER = logging.getLogger(__name__)


class MLflowHelper:
    """Helper to interact with MLflow Tracking and Model Registry.

    The class encapsulates common operations such as starting runs, logging
    parameters or metrics and registering models.  It is purposely lightweight
    to keep a small API surface while still enabling customisation via
    environment variables or constructor arguments.

    Args:
        experiment_name: Name of the experiment to use or create.  If ``None``
            the value of ``MLFLOW_EXPERIMENT`` is used, falling back to
            ``"Default"``.
        default_tags: Tags applied to every run started by this helper.
        tracking_uri: Optional MLflow tracking URI.  When ``None`` the helper
            reads ``MLFLOW_TRACKING_URI`` from the environment or defaults to
            ``"databricks"`` which automatically enables the Databricks
            tracking backend.
        registry_uri: Optional MLflow registry URI.  When ``None`` the tracking
            URI is reused.
        max_retries: Number of times an operation should be retried when a
            transient error occurs.
        retry_wait: Initial waiting time in seconds between retries.  The delay
            grows exponentially with a factor of two.

    Raises:
        ValueError: If ``max_retries`` is lower than ``1``.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        default_tags: Optional[Dict[str, str]] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        *,
        max_retries: int = 3,
        retry_wait: float = 1.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.experiment_name = (
            experiment_name or os.getenv("MLFLOW_EXPERIMENT", "Default")
        )
        self.default_tags = default_tags or {}
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "databricks"
        )
        self.registry_uri = registry_uri or self.tracking_uri
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.experiment_id: Optional[str] = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _call_with_retries(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute ``func`` retrying on failure.

        Args:
            func: Callable to execute.
            *args: Positional arguments passed to ``func``.
            **kwargs: Keyword arguments passed to ``func``.

        Returns:
            Any: Whatever ``func`` returns.

        Raises:
            Exception: Propagates the last exception raised by ``func`` once all
            retries are exhausted.
        """

        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - generic retry
                attempt += 1
                if attempt >= self.max_retries:
                    LOGGER.error("Operation failed after %s attempts", attempt)
                    raise
                sleep_time = self.retry_wait * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Operation failed (%s), retrying in %.1fs", exc, sleep_time
                )
                time.sleep(sleep_time)

    def _ensure_experiment(self) -> str:
        """Return the experiment ID, creating the experiment if necessary."""

        exp = self._call_with_retries(
            mlflow.get_experiment_by_name, self.experiment_name
        )
        if exp is None:
            exp_id = self._call_with_retries(
                mlflow.create_experiment, self.experiment_name
            )
        else:
            exp_id = exp.experiment_id
        self.experiment_id = exp_id
        return exp_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure(self) -> None:
        """Configure MLflow and ensure the experiment exists."""

        LOGGER.debug("Configuring MLflow: tracking=%s registry=%s", self.tracking_uri, self.registry_uri)
        self._call_with_retries(mlflow.set_tracking_uri, self.tracking_uri)
        self._call_with_retries(mlflow.set_registry_uri, self.registry_uri)
        self._ensure_experiment()

    @contextmanager
    def run(
        self,
        run_name: str,
        *,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager to start and finish an MLflow run.

        Args:
            run_name: Name assigned to the run.
            nested: When ``True`` the run is marked as nested.
            tags: Additional tags appended to ``default_tags``.

        Examples:
            >>> helper = MLflowHelper("demo")
            >>> helper.configure()
            >>> with helper.run("train"):
            ...     helper.log_params({"lr": 0.1})
        """

        self.start_run(run_name, nested=nested, tags=tags)
        try:
            yield
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run()

    # ------------------------------------------------------------------
    def start_run(
        self,
        run_name: str,
        *,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Start an MLflow run.

        Args:
            run_name: Name assigned to the run.
            nested: When ``True`` the run is started as nested run.
            tags: Additional tags appended to ``default_tags``.
        """

        all_tags = dict(self.default_tags)
        if tags:
            all_tags.update(tags)
        with self._lock:
            self._call_with_retries(
                mlflow.start_run,
                run_name=run_name,
                experiment_id=self.experiment_id,
                nested=nested,
                tags=all_tags,
            )

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Terminate the active MLflow run."""

        with self._lock:
            self._call_with_retries(mlflow.end_run, status=status)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the active run."""

        self._call_with_retries(mlflow.log_params, params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Mapping of metric names to values.
            step: Optional training step.
        """

        self._call_with_retries(mlflow.log_metrics, metrics, step=step)

    def log_artifacts(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log a directory of artifacts."""

        self._call_with_retries(
            mlflow.log_artifacts, path, artifact_path=artifact_path
        )

    def log_figure(self, fig: Any, artifact_file: str) -> None:
        """Log a matplotlib figure or plotly figure."""

        self._call_with_retries(mlflow.log_figure, fig, artifact_file)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the active run."""

        self._call_with_retries(mlflow.set_tags, tags)

    def enable_autolog(self, frameworks: Sequence[str]) -> None:
        """Enable MLflow autologging for the provided frameworks.

        Args:
            frameworks: Iterable of framework identifiers, e.g. ``("sklearn",)``.
        """

        for fw in frameworks:
            try:
                module = getattr(mlflow, fw)
            except AttributeError:  # pragma: no cover - depends on installed libs
                LOGGER.debug("Framework %s not supported by MLflow", fw)
                continue
            try:
                self._call_with_retries(module.autolog)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Autolog for %s failed: %s", fw, exc)

    def register_model(self, model_uri: str, name: str) -> Any:
        """Register a model in MLflow Model Registry."""

        return self._call_with_retries(mlflow.register_model, model_uri, name)

    def log_and_register_model(
        self,
        model: Any,
        name: str,
        *,
        flavor: str = "pyfunc",
        flavor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Log a model artifact and register it.

        Args:
            model: Model object to log.
            name: Registered model name.
            flavor: MLflow flavor to use (e.g., ``"sklearn"``, ``"xgboost"`` or
                ``"pyfunc"``).
            flavor_kwargs: Additional kwargs passed to the underlying log
                function.
        """

        flavor_kwargs = flavor_kwargs or {}
        if flavor == "pyfunc":
            log_fn = mlflow.pyfunc.log_model
            kwargs = {"python_model": model, **flavor_kwargs}
        else:
            log_module = getattr(mlflow, flavor)
            log_fn = log_module.log_model
            kwargs = {"model": model, "artifact_path": "model", **flavor_kwargs}
        self._call_with_retries(log_fn, **kwargs)
        run = mlflow.active_run()
        if run is None:  # pragma: no cover - defensive
            raise RuntimeError("No active run found for model logging")
        model_uri = f"runs:/{run.info.run_id}/model"
        return self.register_model(model_uri, name)
