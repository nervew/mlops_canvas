# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# Raíz del repo: .../mlops_canvas
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQ_FILE     = PROJECT_ROOT / "requirements.txt"
LOGS_DIR     = PROJECT_ROOT / "logs"
CACHE_FILE   = LOGS_DIR / "requirements.sha1"

# Paquetes “sentinela” para verificar que el entorno quedó usable
# (no hacemos import profundo que pueda romper por ABI; solo import base)
SENTINELS: List[Tuple[str, str]] = [
    ("numpy", "np.__version__"),
    ("pandas", "pd.__version__"),
    ("sklearn", "sklearn.__version__"),
    ("xgboost", "xgboost.__version__"),
    ("lightgbm", "lightgbm.__version__"),
    ("onnx", "onnx.__version__"),
    ("onnxruntime", "ort.get_available_providers()"),  # import onnxruntime as ort
    ("pyarrow", "pa.__version__"),
    ("sweetviz", "sv.__version__"),
]


def _hash_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_cached_hash() -> str | None:
    if CACHE_FILE.exists():
        try:
            return CACHE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def _write_cached_hash(h: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(h, encoding="utf-8")


def _imports_ok() -> bool:
    """
    Intenta importar un conjunto de librerías clave. No imprime nada.
    Devuelve True si todos importan sin excepción.
    """
    try:
        import importlib  # noqa: F401

        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        import lightgbm  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime as ort  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import sweetviz as sv  # noqa: F401

        # Accesos ligeros (evitan fallos silenciosos de ABI)
        _ = np.__version__
        _ = pd.__version__
        _ = sklearn.__version__
        _ = xgboost.__version__
        _ = lightgbm.__version__
        _ = onnx.__version__
        _ = ort.get_available_providers()
        _ = pa.__version__
        _ = sv.__version__
        return True
    except Exception:
        return False


def _pip_install(requirements: Path, force_reinstall: bool = False) -> None:
    """
    Ejecuta `pip install -r requirements.txt` sin '--user'.
    """
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    if force_reinstall:
        cmd.insert(4, "--force-reinstall")  # después de 'install'
    # Passthrough de salida de pip:
    subprocess.run(cmd, check=True)


def install_requirements(force_reinstall: bool = False) -> None:
    """
    Orquesta la instalación con caché:
      1) Si no hay requirements.txt → error claro.
      2) Si hash no cambió y los imports funcionan → no instala.
      3) Si cambió o falla un import → instala (sin '--user').
      4) Verifica imports; si fallan, lanza error.
    """
    print("\nMÓDULO m00 • Instalación de dependencias")
    print("[0.1] Instalando/verificando dependencias…")

    if not REQ_FILE.exists():
        raise FileNotFoundError(f"No se encontró {REQ_FILE}")

    print(f"[0.2] requirements: {REQ_FILE.name}")

    current_hash = _hash_file(REQ_FILE)
    cached_hash  = _read_cached_hash()

    need_install = force_reinstall or (current_hash != cached_hash) or (not _imports_ok())

    if not need_install:
        print("[0.3] Dependencias ya instaladas (caché vigente).")
        return

    print("[0.3] Instalando paquetes de requirements.txt…")
    try:
        _pip_install(REQ_FILE, force_reinstall=force_reinstall)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fallo ejecutando pip (código {e.returncode}).") from e

    # Verificación post-instalación
    if not _imports_ok():
        raise RuntimeError("Instalación completó pero no se pudieron importar paquetes sentinela.")

    _write_cached_hash(current_hash)
    print("[0.4] Instalación completada y caché actualizado ✅")
