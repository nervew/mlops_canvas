# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable

PY_VER = f"python{sys.version_info.major}.{sys.version_info.minor}"

# --- PINS compatibles con sklearn 1.4.x y onnxruntime 1.18.x ---
PINS_SCI = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "pandas==2.2.3",
    "matplotlib==3.8.4",
    "pyarrow==21.0.0",
    "joblib==1.5.2",
    "threadpoolctl==3.6.0",
    "scikit-learn==1.4.2",
]

PINS_ONNX = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "scikit-learn==1.4.2",
    "pybind11==3.0.1",
    "onnx==1.16.0",
    "onnxruntime==1.18.1",
    "onnxmltools==1.13.0",
    "onnxconverter-common==1.16.0",
    "skl2onnx==1.19.1",
    "protobuf==6.32.1",
    "flatbuffers==25.2.10",
    "coloredlogs==15.0.1",
    "humanfriendly==10.0",
    "sympy==1.14.0",
    "packaging==25.0",
]

PINS_RESTO = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "scikit-learn==1.4.2",
    "xgboost==1.7.6",
    "lightgbm==4.6.0",
    "catboost==1.2.8",
    "flaml[automl]==2.1.2",
    "mljar-supervised==1.0.2",
    "scikit-plot==0.3.7",
    "dtreeviz==2.2.2",
    "shap==0.48.0",
    "category-encoders==2.6.3",
    "seaborn==0.13.2",
    "wordcloud==1.9.4",
    "optuna==3.6.1",
    "pyspark==4.0.1",
    "py4j==0.10.9.9",
    "SQLAlchemy==2.0.43",
    "sweetviz==2.3.1",
    "openml==0.15.1",
    "minio==7.2.16",
    "xmltodict==1.0.2",
    "liac-arff==2.5.0",
    "graphviz==0.21",
    "colour==0.1.5",
    "jinja2==3.1.5",
    "tqdm==4.67.1",
]


def _our_site_packages() -> Path:
    """Ruta site-packages donde pip instala para el intérprete de cluster_libraries."""
    exe = Path(sys.executable)
    # .../python/bin/python -> .../python/lib/pythonX.Y/site-packages
    return exe.parent.parent / "lib" / PY_VER / "site-packages"


def _pip(args: Iterable[str]) -> None:
    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--no-cache-dir", "--disable-pip-version-check", "--force-reinstall", *args
    ]
    print(f"[m00] Ejecutando: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _prepend_our_site_packages() -> None:
    """
    Pone nuestra ruta al inicio de sys.path para que gane precedencia sobre
    /databricks/python/... (donde suele vivir NumPy 2.x en DBR 17.x).
    """
    sp = str(_our_site_packages())
    if sp not in sys.path:
        sys.path.insert(0, sp)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")  # evita user-site del driver


def _ensure_numpy_1x() -> None:
    """
    Si NumPy ya fue importado, lo recarga desde nuestra ruta y valida 1.26.x.
    Debe llamarse ANTES de importar pandas/sklearn/matplotlib en el pipeline.
    """
    import importlib
    if "numpy" in sys.modules:
        del sys.modules["numpy"]
    importlib.invalidate_caches()
    _prepend_our_site_packages()
    import numpy as _np  # noqa: F401
    print(f"[m00] NumPy activo: {_np.__version__}", flush=True)
    if not _np.__version__.startswith("1.26"):
        raise RuntimeError(
            "Se está usando un NumPy distinto a 1.26.x. Revisa el orden de imports "
            "y la precedencia de sys.path (nuestra site-packages debe ir primero)."
        )


def install_requirements(requirements_path: str | None = None) -> None:
    """
    Instala paquetes "pinneados" y garantiza que la sesión importe NumPy 1.26.x.
    Si pasas `requirements_path`, también instala ese requirements.txt al final.
    """
    print("[m00] Instalando núcleo científico + scikit-learn (pins)…", flush=True)
    _pip(PINS_SCI)

    print("\n[m00] Instalando stack ONNX (reafirma pins)…", flush=True)
    _pip(PINS_ONNX)

    print("\n[m00] Instalando resto de dependencias (mljar/s-plot)…", flush=True)
    _pip(PINS_RESTO)

    if requirements_path:
        req = Path(requirements_path)
        if req.exists():
            print(f"\n[m00] Instalando extras de {req}…", flush=True)
            _pip(["-r", str(req)])

    # Priorizar nuestra site-packages y fijar NumPy 1.26.x activo
    _prepend_our_site_packages()
    _ensure_numpy_1x()

    # Señales útiles
    try:
        import onnxruntime as ort  # noqa
        print(f"[m00] onnxruntime OK: {ort.__version__}", flush=True)
    except Exception as e:
        print(f"[m00] Advertencia: onnxruntime no se pudo importar: {e}", flush=True)
