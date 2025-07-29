# __init__.py
from importlib import import_module as _imp

def run_pipeline(*args, **kwargs):
    return _imp(__name__ + ".pipeline_reduction").run_feature_reduction(*args, **kwargs)

__all__ = ["run_pipeline"]
