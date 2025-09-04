# app/prueba_m08_backtest.py
from _future_ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort

# Rutas relativas al repo
HERE = Path(_file_).resolve()
PROJECT_ROOT = HERE.parents[2]  # .../mlops_canvas
PART_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
TRANSF_FINAL = PROJECT_ROOT / "transformers" / "transformador_final.onnx"
OUT_DIR = PROJECT_ROOT / "output" / "m08_backtest"

PRED_CSV = OUT_DIR / "predicciones_backtest.csv"
PRED_PNG = OUT_DIR / "pred_vs_real_backtest.png"

TARGET = "target"  # nombre estándar de tu etiqueta

def _feed_from_dataframe(sess: ort.InferenceSession, X: pd.DataFrame) -> dict:
    """
    onnxruntime recomienda alimentar por NOMBRE de entrada;
    cada input suele ser un tensor [N, 1]. Construimos el feed dict
    según el tipo esperado por el modelo. Ver:
    - API onnxruntime (Python): sesiones, inputs/outputs
    - Ejemplo sklearn-onnx: pasar un dict {col: values.reshape(-1,1)}
    """
    need = [i.name for i in sess.get_inputs()]
    missing = [c for c in need if c not in X.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas por el ONNX final: {missing}")

    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        t = inp.type.lower()
        col = X[name]
        if "tensor(string)" in t:
            arr = col.astype(object).values.reshape(-1, 1)
        elif "tensor(double)" in t:
            arr = col.to_numpy(dtype=np.float64).reshape(-1, 1)
        else:  # por defecto: float
            arr = col.to_numpy(dtype=np.float32).reshape(-1, 1)
        feed[name] = arr
    return feed

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Cargar backtest
    back = pd.read_parquet(PART_DIR / "backtest_df.parquet")
    print(f"[1] backtest_df: {back.shape}  cols={list(back.columns)}")

    # 2) Separar y (si existe) y preparar X crudo
    y_true = back[TARGET].copy() if TARGET in back.columns else None
    X = back.drop(columns=[TARGET]) if TARGET in back.columns else back.copy()

    # 3) Cargar transformador_final.onnx (ya compone: transformador inicial + selección + modelo)
    if not TRANSF_FINAL.exists():
        raise FileNotFoundError(f"No existe {TRANSF_FINAL}")
    sess = ort.InferenceSession(str(TRANSF_FINAL), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    print(f"[2] ONNX final: {TRANSF_FINAL.name}")
    print(f"    • Entradas esperadas: {input_names}")

    # 4) Inferencia
    feed = _feed_from_dataframe(sess, X)
    y_pred = sess.run(None, feed)[0].ravel()
    print(f"[3] Predicciones calculadas: n={len(y_pred)}")

    # 5) Exportar CSV y, si hay y_true, métricas y gráfica
    out_df = pd.DataFrame({"y_pred": y_pred})
    if y_true is not None:
        out_df.insert(0, "y_true", y_true.values)

        # RMSE robusto a versión de sklearn (>=1.4 root_mean_squared_error)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        try:
            from sklearn.metrics import root_mean_squared_error
            rmse = float(root_mean_squared_error(y_true, y_pred))
        except Exception:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        mae = float(mean_absolute_error(y_true, y_pred))
        print(f"[4] Métricas backtest: MAE={mae:.6f} | RMSE={rmse:.6f}")

        # Gráfica y_true vs y_pred (misma figura)
        plt.figure(figsize=(10, 5))
        plt.plot(y_true.values, label="y_true")
        plt.plot(y_pred, label="y_pred")
        plt.title("Predicción vs Real (backtest) — transformador_final.onnx")
        plt.xlabel("Fila")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PRED_PNG)
        plt.close()
        print(f"[5] Gráfico guardado en {PRED_PNG}")
    else:
        print("[4] No hay columna 'target' en backtest; sólo se exportan predicciones.")

    out_df.to_csv(PRED_CSV, index=False)
    print(f"[6] Predicciones guardadas en {PRED_CSV}")
    print("OK")

if _name_ == "_main_":
    main()