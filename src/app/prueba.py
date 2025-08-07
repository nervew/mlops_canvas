#!/usr/bin/env python3
"""
show_spark_catalog.py

Script para inspeccionar todo lo que hay en spark_catalog:
  • Lista de catálogos
  • Para spark_catalog:
      – Todos los schemas
      – Todas las tablas dentro de cada schema
"""

import sys
from pathlib import Path
from pyspark.sql import SparkSession


def ensure_src_in_path():
    # Garantizar que 'src/' esté en sys.path para imports relativos si hiciera falta
    project_root = Path(__file__).resolve().parents[1]  # mlops_canvas/src
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main():
    ensure_src_in_path()

    spark = (
        SparkSession.builder
        .appName("inspect_spark_catalog")
        .getOrCreate()
    )

    print("\n=== SHOW CATALOGS ===")
    catalogs = spark.sql("SHOW CATALOGS")
    catalogs.show(truncate=False)

    # Nos enfocamos en spark_catalog
    print("\n=== USE CATALOG spark_catalog ===")
    spark.sql("USE CATALOG spark_catalog")

    print("\n=== SHOW SCHEMAS IN spark_catalog ===")
    schemas = spark.sql("SHOW SCHEMAS")
    schemas.show(truncate=False)

    # Para cada schema, listar tablas
    schema_list = [row.databaseName for row in schemas.collect()]
    for schema in schema_list:
        print(f"\n=== SHOW TABLES IN {schema} ===")
        tables = spark.sql(f"SHOW TABLES IN {schema}")
        tables.show(truncate=False)


if __name__ == "__main__":
    main()
