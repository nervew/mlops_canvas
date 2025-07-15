"""
Script CLI para seleccionar el mejor modelo usando FLAMLWrapper.
Invocación mínima (ejemplo):
    python run_model_selector.py --train path_train.csv --test path_test.csv --target target_col
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# ------------------------------------------------------------------
#  Utils (sin dependencias de melitk / génesis para hacerlo autónomo)
# ------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg)


def validate_params(params: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if not params.get(k)]
    if missing:
        for key in missing:
            _log(f"❌ Missing required parameter: '{key}'.")
        sys.exit(1)


# ------------------------------------------------------------------
#  AutoML Wrapper
# ------------------------------------------------------------------
from flaml_wrapper import FLAMLWrapper          # importa del mismo directorio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="CSV con set de entrenamiento")
    parser.add_argument("--test", required=True, help="CSV con set de prueba")
    parser.add_argument("--target", required=True, help="Nombre de la columna objetivo")
    args = parser.parse_args()

    # Carga de datos
    df_train = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)

    target = args.target
    if target not in df_train.columns or target not in df_test.columns:
        _log(f"❌ Columna '{target}' no encontrada en train/test.")
        sys.exit(1)

    X_train, y_train = df_train.drop(columns=[target]), df_train[target]
    X_test, y_test = df_test.drop(columns=[target]), df_test[target]

    selector = FLAMLWrapper(time_budget=300)
    selector.fit(X_train, y_train, X_test, y_test)

    ranking = selector.get_model_ranking()
    best_model_name, best_model_obj = selector.get_best_model()
    best_params = selector.get_best_params()

    _log("📊  Model ranking:")
    _log(ranking.to_string(index=False))
    _log(f"✅  Selected model: {best_model_name}")
    _log(f"🔧  Best params: {best_params}")


if __name__ == "__main__":
    main()
