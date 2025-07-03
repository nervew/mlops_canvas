import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from pipeline import run


def test_full_pipeline_execution():
    """Pipeline runs end-to-end without raising exceptions."""
    try:
        run()
    except Exception as exc:  # pragma: no cover - we just assert no exception
        pytest.fail(f"Pipeline failed: {exc}")

