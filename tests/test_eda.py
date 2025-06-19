import pandas as pd

from eda_univariado.core.stats import EDAUnivariado


def test_eda_basic(tmp_path):
    df = pd.DataFrame({"col1": [1, 2, 3, 100], "col2": ["a", "a", "b", "b"]})
    cfg = {
        "variables": {"numericas": ["col1"], "categoricas": ["col2"]},
        "outliers": {"metodo": "iqr", "factor": 1.5},
        "paths": {"output_report": str(tmp_path)},
        "graficos": {"hist": {"bins": 10, "kde": False}},
    }
    eda = EDAUnivariado(cfg)
    report = eda.run(df)
    assert report["outliers"]["col1"]["num_outliers"] == 1
    assert "col2" in report["stats"]
