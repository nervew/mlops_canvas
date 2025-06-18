"""Training orchestration script."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.adapters.file_source import CSVDataSource
from ml_project.infrastructure.data_ingestion import ingest_data
from ml_project.domain.data_processing import clean_data
from ml_project.domain.features import add_domain_features
from ml_project.domain.model import train_model
from ml_project.infrastructure.model_persistence import save_model


def main() -> None:
    source = CSVDataSource("data/sample.csv")
    df = ingest_data(source)
    df = clean_data(df)
    df = add_domain_features(df)
    y = df["target"]
    X = df.drop(columns=["target"])
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, params = train_model(X_train.values, y_train.values)
    model.fit(X_train, y_train)
    save_model(model, "models/model.pkl")


if __name__ == "__main__":
    main()
