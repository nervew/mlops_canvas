from data_ingestion.application.service import ingest
from eda_univariado.application import service as eda_uni
from data_validation.application import service as validate
from feature_engineering.application import service as fe
from feature_selection.application import service as fs
from model_search.application import service as ms
from hyperparam_search.application import service as hs
from feature_reduction.application import service as fr
from model_creation.application import service as mc
from inference.application import service as infer
from inference_validation.application import service as infer_val
from validate_model.application import service as val_model
from eda_bivariado.application import service as eda_bi
from eda_scores.application import service as eda_scores
from app.split_dataset.robust_data_splitter import RobustDataSplitter
import pandas as pd


def run() -> None:
    # Ingest data
    dataset = ingest()
    df = dataset.data

    # Univariate EDA
    uni_report = eda_uni.run(df)
    print(uni_report.description.head())

    # Validation
    val_report = validate.run(df)
    if not val_report.success:
        raise ValueError("Data validation failed")

    # Split dataset
    splitter = RobustDataSplitter(
        df,
        split_method="random",
        target_column="target",
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1,
    )
    train_df, test_df, _ = splitter.split_data()

    # Feature engineering
    transformer = fe.run(train_df).transformer
    X_train = transformer.fit_transform(train_df.drop(columns=["target"]))
    X_test = transformer.transform(test_df.drop(columns=["target"]))
    y_train = train_df["target"]
    y_test = test_df["target"]

    # Feature selection
    selector = fs.run(pd.DataFrame(X_train), y_train).selector
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    # Model search
    grid_base = ms.run(pd.DataFrame(X_train_sel), y_train).grid

    # Hyperparameter search
    grid = hs.run(pd.DataFrame(X_train_sel), y_train).grid
    best_model = grid.best_estimator_
    best_model.fit(X_train_sel, y_train)

    # Feature reduction
    reducer = fr.run(pd.DataFrame(X_train_sel)).reducer
    X_train_red = reducer.transform(X_train_sel)
    X_test_red = reducer.transform(X_test_sel)

    # Create model
    trained = mc.run(best_model, pd.DataFrame(X_train_red))

    # Inference
    preds = infer.run(trained.model, pd.DataFrame(X_test_red)).values

    # Validate inference
    infer_report = infer_val.run(preds)
    if not infer_report.success:
        raise ValueError("Inference validation failed")

    # Validate model
    metrics = val_model.run(trained.model, pd.DataFrame(X_test_red), y_test)
    print(f"Test accuracy: {metrics.accuracy:.3f}")

    # Bivariate EDA
    eda_bi_report = eda_bi.run(train_df)
    print(eda_bi_report.correlation.head())

    # Score analysis
    score_report = eda_scores.run(y_test.to_numpy(), preds.to_numpy())
    print(score_report.metrics)


if __name__ == "__main__":
    run()
