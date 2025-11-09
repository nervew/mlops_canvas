"""Telemetry utilities leveraging OpenTelemetry and Azure Monitor."""
from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.azuremonitor import AzureMonitorTraceExporter

logger = logging.getLogger(__name__)


def configure_tracing(connection_string: str) -> None:
    resource = Resource.create({"service.name": "mlops-online"})
    provider = TracerProvider(resource=resource)
    exporter = AzureMonitorTraceExporter.from_connection_string(connection_string)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("Tracing configured")
