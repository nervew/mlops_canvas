from __future__ import annotations

from pathlib import Path
import yaml

from eda_univariado.adapters.pandas_loader import PandasLoader
from eda_univariado.adapters.json_writer import JsonWriter
from eda_univariado.core.stats import EDAUnivariado


if __name__ == "__main__":
    with open(
        Path("src/eda_univariado/config/config.yaml"), "r", encoding="utf-8"
    ) as fh:
        cfg = yaml.safe_load(fh)

    loader = PandasLoader(Path(cfg["paths"]["input"]))
    data_raw = loader.load()

    eda = EDAUnivariado(cfg)
    report = eda.run(data_raw)

    writer = JsonWriter(Path(cfg["paths"]["output_report"]) / "report.json")
    writer.write(report)
