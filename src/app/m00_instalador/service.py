# src/app/m00_instalador/service.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

def install_requirements(req_filename: str = "requirements.txt") -> None:
    """
    Instalador simple y determinista para Databricks/terminal:
      - pip install -r requirements.txt (sin --target)
      - smoke-test de imports clave (incluye sweetviz)
    """
    # .../src/app/m00_instalador/service.py → parents[3] = .../mlops_canvas
    project_root = Path(__file__).resolve().parents[3]
    req_file = project_root / req_filename
    if not req_file.exists():
        raise FileNotFoundError(f"No existe el archivo de requerimientos: {req_file}")

    print("[m00] Instalando/verificando dependencias fijadas…", flush=True)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--no-cache-dir", "--disable-pip-version-check",
        "-r", str(req_file),
    ]
    # Fallar pronto si pip falla
    subprocess.run(cmd, check=True)

    # Smoke-test: si algo falta, ImportError aquí evita errores tardíos
    for mod in ("numpy", "pandas", "sklearn", "onnxruntime", "sweetviz"):
        __import__(mod)

    print("[m00] Dependencias listas (requisitos satisfechos) ✅", flush=True)
