"""
Instala automáticamente las dependencias del proyecto.

Uso:
    >>> from app.m00_instalador import install_requirements
    >>> install_requirements()                 # requirements.txt por defecto
    >>> install_requirements("otro.txt")       # ruta específica
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _default_requirements_path() -> Path:
    """
    Devuelve la ruta …/requirements.txt a partir de este archivo.
    La estructura esperada es:

      proyecto/
      ├── requirements.txt
      └── src/
          └── app/
              └── instalador/
                  └── service.py  (este archivo)
    """
    # …/src/app/instalador/service.py  -> parents[3]  ≡  proyecto/
    return Path(__file__).resolve().parents[3] / "requirements.txt"


def install_requirements(requirements_path: str | Path | None = None) -> None:
    """Instala los paquetes listados en *requirements_path* usando pip."""
    req_path = Path(requirements_path) if requirements_path else _default_requirements_path()

    if not req_path.exists():
        # No abortamos: simplemente avisamos y continuamos.
        print(f"⚠️  requirements.txt no encontrado en: {req_path}")
        return

    print(f"📦 Instalando dependencias desde {req_path} …", flush=True)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Muestra el log de pip
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Error instalando requirements.txt:\n", result.stderr, flush=True)
    else:
        print("✅ Todas las dependencias están instaladas.", flush=True)
