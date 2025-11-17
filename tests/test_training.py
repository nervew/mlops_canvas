import pandas as pd
from sklearn.datasets import make_classification

from training_api.app.automl import AutoMLRunner
from training_api.app.io import load_dataset


def test_detect_task_and_pipeline():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    runner = AutoMLRunner(time_budget=1)
    result, pipeline = runner.run(df.drop(columns=["target"]), df["target"])
    assert result.estimator
    assert pipeline is not None
