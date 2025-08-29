# MLflow Helper

This module provides the `MLflowHelper` class, a light abstraction over the
MLflow APIs tailored for Databricks. The helper centralises configuration of
the tracking and registry URIs, handles experiment creation, and offers a
context manager for runs.

```python
from mlops.mlflow_helper import MLflowHelper

helper = MLflowHelper(
    experiment_name="MyExperiment",
    default_tags={"project": "demo"}
)
helper.configure()

with helper.run("train", tags={"stage": "training"}):
    helper.log_params({"lr": 0.01})
    helper.log_metrics({"mae": 0.5}, step=1)
```

The helper reads credentials from the environment and never prints secrets. It
supports retries for transient failures and optional autologging for common
frameworks such as scikit-learn or XGBoost.
