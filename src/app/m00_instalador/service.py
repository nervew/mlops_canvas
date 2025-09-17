"""
Instalador robusto para Databricks:
- Fija NumPy 1.26.4 y XGBoost 1.7.6 con constraints.
- Instala el stack ONNX compatible.
- Verifica onnxruntime en un SUBPROCESO para evitar conflictos del intérprete actual.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Pines de compatibilidad
NUMPY_PIN = "numpy==1.26.4"
XGB_PIN   = "xgboost==1.7.6"
ONNX_STACK = [
    "pybind11>=2.12",
    "onnx==1.16.0",
    "onnxruntime==1.18.1",
    "onnxmltools==1.13.0",
    "onnxconverter-common==1.16.0",
    "skl2onnx==1.19.1",
]

BASE_FLAGS = ["--upgrade", "--no-cache-dir"]


# -------------------- utilidades --------------------
def _default_requirements_path() -> Path:
    # …/src/app/m00_instalador/service.py -> proyecto/
    return Path(__file__).resolve().parents[3] / "requirements.txt"

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[m00] Ejecutando: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)

def _pip_install(pkgs: list[str], force: bool = False, no_deps: bool = False) -> None:
    if not pkgs:
        return
    args = [sys.executable, "-m", "pip", "install", *BASE_FLAGS]
    if force:
        args.append("--force-reinstall")
    if no_deps:
        args.append("--no-deps")
    proc = _run(args + pkgs)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"[m00] pip falló instalando: {pkgs}")

def _write_constraints(tmp_dir: Path) -> Path:
    """
    Crea un constraints.txt para impedir que requirements.txt vuelva a subir NumPy/XGBoost.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cpath = tmp_dir / "constraints.txt"
    cpath.write_text("\n".join([NUMPY_PIN, XGB_PIN]) + "\n", encoding="utf-8")
    return cpath

def _verify_onnx_in_subprocess() -> None:
    """
    Verifica NumPy+onnxruntime en un proceso limpio (evita NumPy ya importado=2.x).
    No levanta excepción si falla: sólo informa (m08/ONNX lo volverán a intentar).
    """
    code = (
        "import numpy as np; "
        "import onnxruntime as ort; "
        "print(f'NumPy={np.__version__} | onnxruntime={ort.__version__} | device={ort.get_device()}')"
    )
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if proc.returncode == 0:
        print("[m00] Verificación OK →", proc.stdout.strip(), flush=True)
    else:
        # No abortamos para no romper el pipeline; sólo informamos el motivo.
        print("[m00] Aviso: onnxruntime no pudo importarse en verificación aislada.", flush=True)
        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, flush=True)


# -------------------- instalación principal --------------------
def install_requirements(requirements_path: str | Path | None = None) -> None:
    # 0) constraints para bloquear NumPy/XGB
    constraints = _write_constraints(Path("/tmp/m00_instalador"))

    # 1) requirements.txt (si existe) con constraints
    req_path = Path(requirements_path) if requirements_path else _default_requirements_path()
    if req_path.exists():
        print(f"📦 Instalando dependencias desde {req_path} (con constraints)…", flush=True)
        proc = _run([sys.executable, "-m", "pip", "install", *BASE_FLAGS, "-r", str(req_path),
                     "-c", str(constraints)])
        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.returncode != 0:
            print("❌ Error instalando requirements.txt:\n" + proc.stderr, file=sys.stderr, flush=True)
            # seguimos con los pines críticos igualmente
    else:
        print(f"⚠️  requirements.txt no encontrado en: {req_path}", flush=True)

    # 2) Reforzar pines críticos (sin dependencias para que no re-pisen NumPy)
    _pip_install([NUMPY_PIN], force=True, no_deps=True)
    _pip_install([XGB_PIN],   force=False, no_deps=True)

    # 3) ONNX stack compatible (reinstalación forzada para asegurar binarios correctos)
    _pip_install(ONNX_STACK, force=True)

    # 4) Verificación en SUBPROCESO (no en este intérprete)
    _verify_onnx_in_subprocess()

    # 5) Info de entorno
    print(f"[m00] Python: {sys.executable}", flush=True)
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        print(f"[m00] Databricks Runtime: {os.environ['DATABRICKS_RUNTIME_VERSION']}", flush=True)
    print("✅ Dependencias listas.\n", flush=True)
