import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from data_ingestion.application.service import ingest
from data_validation.application.service import run as validate_run
from feature_engineering.application.service import run as fe_run
from feature_selection.application.service import run as fs_run
from inference.application.service import run as infer_run
from inference_validation.application.service import run as infer_val_run
from validate_model.application.service import run as val_model_run
from model_search.application.service import run as model_search_run
from feature_reduction.application.service import run as fr_run
from model_creation.application.service import run as mc_run


def test_ingest_and_validate():
    dataset = ingest()
    df = dataset.data
    assert not df.empty
    report = validate_run(df)
    assert report.success


def test_feature_engineering_and_selection():
    df = ingest().data
    transformer = fe_run(df).transformer
    X = transformer.fit_transform(df.drop(columns=["target"]))
    selector = fs_run(pd.DataFrame(X), df["target"]).selector
    X_sel = selector.transform(X)
    assert X_sel.shape[0] == df.shape[0]


def test_model_training_and_inference():
    df = ingest().data
    transformer = fe_run(df).transformer
    X = transformer.fit_transform(df.drop(columns=["target"]))
    y = df["target"]
    selector = fs_run(pd.DataFrame(X), y).selector
    X_sel = selector.transform(X)
    grid = model_search_run(pd.DataFrame(X_sel), y).grid
    best_model = grid.best_estimator_
    best_model.fit(X_sel, y)
    reducer = fr_run(pd.DataFrame(X_sel)).reducer
    X_red = reducer.transform(X_sel)
    trained = mc_run(best_model, pd.DataFrame(X_red))
    preds = infer_run(trained.model, pd.DataFrame(X_red)).values
    infer_report = infer_val_run(preds)
    assert infer_report.success
    metrics = val_model_run(trained.model, pd.DataFrame(X_red), y)
    assert metrics.accuracy > 0.5

