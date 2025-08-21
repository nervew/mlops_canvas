from pathlib import Path

import pandas as pd

from app.m02_eda_univariado.application.run import run


def test_run_returns_report_and_calls_generate(monkeypatch, sample_df, tmp_path):
    called = {}

    def fake_generate(df, path):
        called["path"] = Path(path)

    monkeypatch.setattr(
        "app.m02_eda_univariado.application.run.generate_sweetviz_report",
        fake_generate,
    )
    report = run(sample_df, tmp_path / "rep.html")
    assert "num" in report.description
    assert called["path"].name == "rep.html"
