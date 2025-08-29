from __future__ import annotations

"""Utility class to interact with the AWS SageMaker Model Registry.

The :class:`AWSModelRegistryHelper` class offers a minimal, yet convenient,
wrapper around the SageMaker Model Registry API via :mod:`boto3`.  It mirrors
part of the interface of :class:`mlops.mlflow_helper.MLflowHelper` to provide a
consistent developer experience when switching between MLflow and AWS as model
registry backends.

The helper focuses on the **registration** step and intentionally keeps a small
API surface to honour the KISS and YAGNI principles.  It supports automatic
creation of model package groups and includes retry logic for transient errors.
"""

import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

LOGGER = logging.getLogger(__name__)


class AWSModelRegistryHelper:
    """Interact with AWS SageMaker Model Registry.

    Args:
        region_name: AWS region hosting the SageMaker service.
        boto_client: Pre-configured SageMaker client.  When ``None`` a new
            client is created via :func:`boto3.client`.
        max_retries: Maximum number of attempts when a call fails.
        retry_wait: Initial delay between retries in seconds.  The delay grows
            exponentially with a factor of two.

    Raises:
        ValueError: If ``max_retries`` is lower than ``1``.
    """

    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        boto_client: Optional[Any] = None,
        max_retries: int = 3,
        retry_wait: float = 1.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.client = boto_client or boto3.client("sagemaker", region_name=region_name)
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _call_with_retries(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute ``func`` retrying on failure.

        This method mirrors :meth:`mlops.mlflow_helper.MLflowHelper._call_with_retries`
        to keep behaviour consistent across backends.  Any exception raised by
        ``func`` triggers a retry until ``max_retries`` is exhausted.

        Args:
            func: Callable to execute.
            *args: Positional arguments passed to ``func``.
            **kwargs: Keyword arguments passed to ``func``.

        Returns:
            The return value of ``func``.

        Raises:
            Exception: Propagates the last exception once retries are exhausted.
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

    def _ensure_model_package_group(self, name: str, description: str = "") -> None:
        """Create the model package group if it does not already exist.

        Args:
            name: Name of the model package group.
            description: Optional human readable description.
        """

        try:
            self._call_with_retries(
                self.client.describe_model_package_group,
                ModelPackageGroupName=name,
            )
        except ClientError as exc:
            error = exc.response.get("Error", {})
            if error.get("Code") != "ResourceNotFound":
                raise
            self._call_with_retries(
                self.client.create_model_package_group,
                ModelPackageGroupName=name,
                ModelPackageGroupDescription=description,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_model(
        self,
        *,
        model_data_url: str,
        image_uri: str,
        model_package_group_name: str,
        model_package_name: Optional[str] = None,
        description: str = "",
        model_metrics: Optional[Dict[str, Any]] = None,
        approval_status: str = "PendingManualApproval",
    ) -> str:
        """Register a model artifact in SageMaker Model Registry.

        This method uploads metadata to SageMaker linking an S3 artifact with an
        inference image.  Only a subset of the full API is exposed to keep the
        interface compact and avoid unnecessary complexity.

        Args:
            model_data_url: S3 URI pointing to the model artifact.
            image_uri: URI of the container image used for inference.
            model_package_group_name: Destination model package group.
            model_package_name: Optional name for the model package.
            description: Optional description for the package.
            model_metrics: Metrics associated with the model.  Must follow the
                structure expected by SageMaker's ``ModelMetrics``.
            approval_status: Initial approval status, e.g. ``"Approved"`` or
                ``"PendingManualApproval"``.

        Returns:
            The ARN of the created model package.
        """

        with self._lock:
            self._ensure_model_package_group(model_package_group_name)
            request: Dict[str, Any] = {
                "ModelPackageGroupName": model_package_group_name,
                "ModelPackageDescription": description,
                "InferenceSpecification": {
                    "Containers": [
                        {"Image": image_uri, "ModelDataUrl": model_data_url}
                    ],
                    "SupportedContentTypes": ["text/csv"],
                    "SupportedResponseMIMETypes": ["text/csv"],
                },
                "ModelApprovalStatus": approval_status,
            }
            if model_package_name:
                request["ModelPackageName"] = model_package_name
            if model_metrics:
                request["ModelMetrics"] = model_metrics

            response = self._call_with_retries(
                self.client.create_model_package, **request
            )
        return response["ModelPackageArn"]

