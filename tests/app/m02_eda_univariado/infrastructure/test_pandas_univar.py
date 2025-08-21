from pathlib import Path

import pandas as pd

from app.m02_eda_univariado.infrastructure.pandas_univar import (
    run_univariate_analysis,
    generate_sweetviz_report,
)


def test_run_univariate_analysis_numeric_and_cat(sample_df):
    report = run_univariate_analysis(sample_df)
    assert report["num"]["count"] == 4
    assert report["cat"]["unique"] == 2


def test_generate_sweetviz_report(tmp_path, sample_df, monkeypatch):
    output = tmp_path / "rep.html"

    class DummyReport:
        def show_html(self, path, open_browser=False):
            Path(path).write_text("html")

    def fake_analyze(df):
        return DummyReport()

    monkeypatch.setattr(
        "app.m02_eda_univariado.infrastructure.pandas_univar.sv.analyze",
        fake_analyze,
    )
    generate_sweetviz_report(sample_df, output)
    assert output.exists()
