# check_env.py

import sys
import os
import importlib

def main():
    print("=== Entorno de ejecución ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("sys.path (primeros 3):")
    for p in sys.path[:3]:
        print(f"  {p}")

    print("\n=== Librerías críticas ===")
    # NumPy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print(f"NumPy file: {np.__file__}")
    except Exception as e:
        print(f"ERROR al importar NumPy: {e}")

    # onnxruntime
    try:
        import onnxruntime as ort
        print(f"\nonnxruntime version: {getattr(ort, '__version__', 'desconocida')}")
        print(f"onnxruntime file: {ort.__file__}")
    except Exception as e:
        print(f"\nERROR al importar onnxruntime: {e}")

    # Verificar si numpy es mayor o igual que 2
    try:
        from packaging.version import Version
        if 'np' in locals():
            ver = Version(np.__version__)
            if ver.major >= 2:
                print("\n⚠️ Advertencia: NumPy >= 2 detectada; puede causar incompatibilidades con librerías compiladas contra NumPy 1.x")
            else:
                print("\n✅ NumPy < 2: parece compatible")
    except ImportError:
        print("\nAviso: no se puede importar 'packaging' para comparar versiones")

if __name__ == "__main__":
    main()
