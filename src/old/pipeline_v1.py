# ✦ src/pipeline.py ✦
import os
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

# Obtiene la ruta del directorio donde está este script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────
# 1) Ingestión de datos de ejemplo
# ──────────────────────────────
def data_ingestion(n_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_samples, freq="D"),
        "category": rng.choice(["A", "B", "C"], size=n_samples),
        "num1": rng.normal(size=n_samples),
        "num2": rng.uniform(0, 100, size=n_samples),
        "target": rng.integers(0, 2, size=n_samples),
    })
    return df

# ──────────────────────────────
# 2) Split temporal robusto
# ──────────────────────────────
def data_split(df: pd.DataFrame):
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

# ──────────────────────────────
# 3) Feature engineering *consistente*
# ──────────────────────────────
def _base_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day
    df = df.drop(columns=["date"])
    return df

def data_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Aplica transformaciones base
    train = _base_transform(train_df)
    test  = _base_transform(test_df)

    # Concatena para generar dummies coherentes
    combined = pd.concat([train, test], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=["category"], drop_first=False)

    # Separa nuevamente
    train = combined.xs("train")
    test  = combined.xs("test")

    # Elimina columnas constantes (en train)
    constant_cols = train.columns[train.nunique() <= 1]
    train = train.drop(columns=constant_cols)
    test  = test.drop(columns=constant_cols, errors="ignore")

    # Asegura misma columna y orden
    test = test.reindex(columns=train.columns, fill_value=0)

    return train, test

# ──────────────────────────────
# 4) Selección de features
# ──────────────────────────────
def feature_selection(X_train, y_train, X_test):
    selector = SelectKBest(score_func=f_classif, k=min(8, X_train.shape[1]))
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test), selector

# ──────────────────────────────
# 5) Búsqueda de hiperparámetros
# ──────────────────────────────
def search_model(X_train, y_train):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [50, 100], "max_depth": [None, 5]},
        cv=3,
    )
    grid.fit(X_train, y_train)
    return grid

# ──────────────────────────────
# 6) Entrena y guarda versión del modelo
# ──────────────────────────────
def create_model(grid, X_train, y_train, version: str = "v1"):
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    onnx_path   = os.path.join(MODEL_DIR, f"model_{version}.onnx")
    joblib_path = os.path.join(MODEL_DIR, f"model_{version}.joblib")

    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model   = convert_sklearn(best_model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    joblib.dump(best_model, joblib_path)

    print(f"✅ Modelo guardado en:\n   • {joblib_path}\n   • {onnx_path}", flush=True)
    return best_model

# ──────────────────────────────
# 7) Métrica
# ──────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# ──────────────────────────────
# 8) Orquestador
# ──────────────────────────────
def run_pipeline():
    df = data_ingestion()
    train_df, test_df = data_split(df)
    train_df, test_df = data_engineering(train_df, test_df)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test  = test_df.drop(columns=["target"])
    y_test  = test_df["target"]

    X_train_sel, X_test_sel, _ = feature_selection(X_train, y_train, X_test)
    grid  = search_model(X_train_sel, y_train)
    model = create_model(grid, X_train_sel, y_train, version="v1")

    acc = evaluate(model, X_test_sel, y_test)
    print(f"Test accuracy: {acc:.4f}", flush=True)

if __name__ == "__main__":
    run_pipeline()
