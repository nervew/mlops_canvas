import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

from app.split_dataset.robust_data_splitter import RobustDataSplitter


def data_ingestion(n_samples: int = 500) -> pd.DataFrame:
    """Create a toy dataset with categorical, numerical and time columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_samples, freq="D"),
        "category": rng.choice(["A", "B", "C"], size=n_samples),
        "num1": rng.normal(size=n_samples),
        "num2": rng.uniform(0, 100, size=n_samples),
        "target": rng.integers(0, 2, size=n_samples),
    })
    return df


def data_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically using ``RobustDataSplitter``."""
    splitter = RobustDataSplitter(
        df,
        split_method="time",
        time_column="date",
        target_column="target",
        train_size=0.6,
        test_size=0.2,
        backtest_size=0.2,
    )
    train_df, test_df, _ = splitter.split_data()
    return train_df, test_df


def data_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate time based features and one-hot encode categoricals."""
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df = df.drop(columns=["date"])
        df = pd.get_dummies(df, columns=["category"], drop_first=False)
        return df

    return transform(train_df), transform(test_df)


def feature_selection(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Apply univariate feature selection."""
    selector = SelectKBest(score_func=f_classif, k=min(8, X_train.shape[1]))
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test), selector


def search_model(X_train: np.ndarray, y_train: pd.Series) -> GridSearchCV:
    """Simple hyperparameter search over a random forest."""
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid={"n_estimators": [50, 100], "max_depth": [None, 5]}, cv=3)
    grid.fit(X_train, y_train)
    return grid


def create_model(grid: GridSearchCV, X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
    """Train best estimator on full data and export to ONNX."""
    best_model: RandomForestClassifier = grid.best_estimator_
    best_model.fit(X_train, y_train)

    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(best_model, initial_types=initial_type)
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    joblib.dump(best_model, "model.joblib")
    return best_model


def evaluate(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> float:
    """Compute accuracy on the test split."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def run_pipeline() -> None:
    df = data_ingestion()
    train_df, test_df = data_split(df)
    train_df, test_df = data_engineering(train_df, test_df)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    X_train_sel, X_test_sel, _ = feature_selection(X_train, y_train, X_test)
    grid = search_model(X_train_sel, y_train)
    model = create_model(grid, X_train_sel, y_train)
    acc = evaluate(model, X_test_sel, y_test)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_pipeline()
