#!/usr/bin/env python3
# run_ingest.py

import sys
from pathlib import Path
from pyspark.sql import SparkSession

# ——————————————————————————————
# 1) Asegúrate de que tu proyecto está en el PYTHONPATH
# ——————————————————————————————
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ——————————————————————————————
# 2) Importa tu módulo de ingesta
# ——————————————————————————————
from app.m01_data_ingestion import ingest as ingest_data

def run():
    # ——————————————————————————————
    # 3) Crea o recupera la sesión de Spark
    # ——————————————————————————————
    spark = SparkSession.builder \
        .appName("DataIngestion") \
        .getOrCreate()

    # ——————————————————————————————
    # 4) Lista los esquemas (bases de datos) disponibles
    # ——————————————————————————————
    print("Esquemas disponibles en el metastore de Spark:")
    spark.sql("SHOW DATABASES").show(truncate=False)

    # ——————————————————————————————
    # 5) Ejecuta la ingesta de datos
    # ——————————————————————————————
    df = ingest_data()
    print(f"Ingesta completada ({len(df)} filas).")
    print(df.head())

    # ——————————————————————————————
    # 6) Opcional: cierra la sesión si no la necesitas más
    # ——————————————————————————————
    spark.stop()

if __name__ == "__main__":
    run()
