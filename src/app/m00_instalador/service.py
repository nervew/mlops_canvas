"""
Instala automáticamente las dependencias del proyecto y garantiza compatibilidad
NumPy ↔ ONNX Runtime y XGBoost en Databricks.

Uso:
    >>> from app.m00_instalador import install_requirements
    >>> install_requirements()                 # requirements.txt por defecto
    >>> install_requirements("otro.txt")       # ruta específica
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Pines de compatibilidad
NUMPY_PIN = "numpy==1.26.4"
# XGBoost >=1.6 introduce 'callbacks' en .fit(); 1.7.6 es estable y compatible con onnxmltools
XGB_PIN   = "xgboost==1.7.6"

# ONNX stack compatible con NumPy 1.26.x
ONNX_STACK = [
    "pybind11>=2.12",
    "onnx==1.16.0",
    "onnxruntime==1.18.1",
    "onnxmltools==1.13.0",
    "onnxconverter-common==1.16.0",
    "skl2onnx==1.19.1",
]

# Paquetes base (añadimos xgboost/catboost/lightgbm aquí)
BASE_PKGS = [
    "pandas>=2.1,<2.3",
    "scikit-learn>=1.3,<1.5",
    "matplotlib>=3.8,<3.9",
    "pyarrow>=10.0",
    "joblib>=1.2",
    "sqlalchemy",
    "sweetviz",
    "pyspark>=3.3",
    "py4j>=0.10",
    "flaml[automl]>=2.0,<2.2",
    "lightgbm",
    XGB_PIN,
    "catboost>=1.2.5,<1.3",
    "mljar-supervised",
    "abess",
    "shap",
    "openml",
    "graphviz",
    "colour",
]

BASE_FLAGS = ["--upgrade", "--no-cache-dir"]


def _default_requirements_path() -> Path:
    # …/src/app/m00_instalador/service.py  -> parents[3]  ≡  proyecto/
    return Path(__file__).resolve().parents[3] / "requirements.txt"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[m00] Ejecutando: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def _pip_install(pkgs: list[str], force: bool = False) -> None:
    if not pkgs:
        return
    args = [sys.executable, "-m", "pip", "install", *BASE_FLAGS]
    if force:
        args.append("--force-reinstall")
    proc = _run(args + pkgs)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"[m00] pip falló instalando: {pkgs}")


def _numpy_major() -> int:
    try:
        import numpy as _np  # noqa
        return int(_np.__version__.split(".")[0])
    except Exception:
        return 0


def _ensure_stacks() -> None:
    """Asegura NumPy 1.26, ONNX stack compatible y XGBoost>=1.7.6."""
    # 1) requirements.txt (si existe)
    #    *No* abortamos si falla; aplicamos igualmente los pines abajo.
    req_path = _default_requirements_path()
    if req_path.exists():
        proc = _run([sys.executable, "-m", "pip", "install", *BASE_FLAGS, "-r", str(req_path)])
        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.returncode != 0:
            print("❌ Error instalando requirements.txt:\n" + proc.stderr, file=sys.stderr, flush=True)

    # 2) Forzar NumPy 1.26 si no está ya
    if _numpy_major() != 1:
        print(f"[m00] Ajustando NumPy → {NUMPY_PIN}…", flush=True)
        _pip_install([NUMPY_PIN], force=True)

    # 3) Paquetes base (incluye xgboost==1.7.6)
    _pip_install(BASE_PKGS)

    # 4) ONNX stack compatible
    _pip_install(ONNX_STACK)

    # 5) Verificaciones en caliente (este intérprete)
    try:
        import numpy as np
        import onnxruntime as ort
        import xgboost as xgb  # noqa

        _ = ort.get_device()  # fuerza carga extensión C
        print(f"[m00] OK → NumPy {np.__version__} | onnxruntime {ort.__version__} | device={ort.get_device()}",
              flush=True)
    except Exception as e:
        print("[m00] Verificación falló. Reinstalación forzada de NumPy/ONNX/XGBoost…", flush=True)
        _pip_install([NUMPY_PIN] + ONNX_STACK + [XGB_PIN], force=True)
        import onnxruntime as ort  # noqa
        _ = ort.get_device()
        print(f"[m00] OK tras reinstalar → onnxruntime {ort.__version__}", flush=True)


def install_requirements(requirements_path: str | Path | None = None) -> None:
    """Instala dependencias y asegura compatibilidad crítica."""
    # Si el usuario pasa una ruta específica, la usamos; si no, usamos la por defecto
    if requirements_path is not None:
        req_path = Path(requirements_path)
        if req_path.exists():
            proc = _run([sys.executable, "-m", "pip", "install", *BASE_FLAGS, "-r", str(req_path)])
            if proc.stdout:
                print(proc.stdout, flush=True)
            if proc.returncode != 0:
                print("❌ Error instalando requirements.txt:\n" + proc.stderr, file=sys.stderr, flush=True)
        else:
            print(f"⚠️  requirements.txt no encontrado en: {req_path}", flush=True)

    # Asegurar pines y verificación
    _ensure_stacks()

    print(f"[m00] Python: {sys.executable}", flush=True)
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        print(f"[m00] Databricks Runtime: {os.environ['DATABRICKS_RUNTIME_VERSION']}", flush=True)
    print("✅ Dependencias listas.\n", flush=True)
