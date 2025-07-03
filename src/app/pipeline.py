# pipeline.py

import sys
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator

from app.data_ingestion.application.service import ingest
from app.eda_univariado.application import service as eda_uni
from app.data_validation.application import service as validate
from app.data_split.application import service as splitter
from app.search_model.flaml_wrapper import FLAMLWrapper

# Carpeta donde se guardarán los modelos
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, X_sample, version: str = "v1"):
    """
    Guarda el modelo entrenado en formato joblib (siempre) y ONNX (solo si es compatible).
    """
    joblib_path = MODEL_DIR / f"model_{version}.joblib"
    joblib.dump(model, joblib_path)
    print(f"✅ Modelo guardado como {joblib_path}", flush=True)

    # Intentar guardar en ONNX solo si es un estimador de sklearn
    onnx_path = MODEL_DIR / f"model_{version}.onnx"
    if isinstance(model, BaseEstimator):
        try:
            initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            onnx_path.write_bytes(onnx_model.SerializeToString())
            print(f"✅ Modelo ONNX guardado: {onnx_path}", flush=True)
        except Exception as e:
            print(f"⚠️  No se pudo convertir a ONNX: {e}", flush=True)
    else:
        print("⚠️  El modelo no es un estimador estándar de Scikit-learn. Solo se guarda en joblib.", flush=True)


def run():
    # 1. Ingesta de datos
    dataset = ingest()
    df = dataset.data

    # 2. EDA univariado
    eda_report = eda_uni.run(df)
    print("Reporte EDA:", getattr(eda_report, "html_report_path", "no disponible"))

    # 3. Validación de datos
    val_report = validate.run(df, fit_profile=True)
    if not getattr(val_report, "valido", False):
        print("❌ Validación falló:", getattr(val_report, "detalles", {}))
        raise ValueError("Datos no pasan validación")
    else:
        print("✅ Validación exitosa")

    # 4. Split (train/test/backtest)
    splits = splitter.run(
        df,
        split_method="random",
        target_column="target",
        stratify_columns=[],
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1
    )
    train_df = splits.train_df
    test_df = splits.test_df

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # 5. Entrenamiento y búsqueda de modelo automático
    automl = FLAMLWrapper(time_budget=300)
    automl.fit(X_train, y_train, X_test, y_test)

    ranking = automl.get_model_ranking()
    print("Ranking de modelos:\n", ranking)

    best_model_name, best_model_obj = automl.get_best_model()
    print("Mejor modelo:", best_model_name)
    print("Parámetros óptimos:", automl.get_best_params())

    # 6. Guardar modelo entrenado (joblib y, si es posible, onnx)
    save_model(best_model_obj, X_train, version="v1")


if __name__ == "__main__":
    run()
