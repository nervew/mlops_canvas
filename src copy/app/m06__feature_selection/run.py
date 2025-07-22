from __future__ import annotations
from sklearn.linear_model import LinearRegression

from .step01_import import ParquetPartitionLoader2
from .step02_filtering import filter_partitions
from .step03_frame import frame_partitions
from .step04_abess import abess_partitions
from .step05_shap_select import shap_partitions
from .step06_permutation import permutation_partitions
from .step07_export import export_partitions


def run_pipeline():
    # 1. Carga particiones originales (step01)
    loader = ParquetPartitionLoader2()
    X_train, X_test, X_back, y_train, y_test, y_back = loader.load()

    # 2. Filtrado inicial (step02)
    X_train, X_test, X_back, filter_selector = filter_partitions(
        X_train, X_test, X_back, y_train
    )

    # 3. Selección híbrida forward/RFE (step03) usando regresión
    X_train, X_test, X_back, frame_selector = frame_partitions(
        X_train,
        X_test,
        X_back,
        y_train,
        estimator=LinearRegression(),
        forward_k=3,
        final_k=2,
    )

    # 4. ABESS selección (step04) – regresión
    X_train, X_test, X_back, abess_selector = abess_partitions(
        X_train,
        X_test,
        X_back,
        y_train,
        mode="regression",
    )

    # 5. Selección por SHAP (step05) – regresión
    X_train, X_test, X_back, shap_selector = shap_partitions(
        X_train,
        X_test,
        X_back,
        y_train,
        top_n=3,
        task="regression",
    )

    # 6. Selección por Permutación (step06) – regresión
    X_train, X_test, X_back, permutation_selector = permutation_partitions(
        X_train,
        X_test,
        X_back,
        y_train,
        tol=0.01,
        task="regression",
        scoring="r2",
    )

    # 7. Exportación final de particiones seleccionadas (step07)
    export_partitions(X_train, X_test, X_back)

    print("Pipeline finalizado. Variables seleccionadas finales:")
    print(X_train.columns.tolist())

    return X_train, X_test, X_back
