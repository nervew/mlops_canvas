import os
import json
import pandas as pd
from sqlalchemy import create_engine, text

class ConsultaDB:
    """
    Clase de ayuda para:
      • Conectarse a MySQL (opcional).
      • Leer archivos planos (.csv | .parquet | .json | .xlsx).
      • Ejecutar una consulta SQL (.sql) y devolver un DataFrame.
      • Exportar los resultados en varios formatos.
    """

    def __init__(self, config_path: str | None = None):
        base_dir = os.path.dirname(__file__)
        cfg_file = config_path or os.path.join(base_dir, "config.json")

        with open(cfg_file, "r") as f:
            cfg = json.load(f)

        # Parámetros comunes
        self.output_dir = os.path.join(base_dir, cfg.get("output_folder", "salidas"))
        os.makedirs(self.output_dir, exist_ok=True)

        # Conexión a base de datos (opcional)
        self.use_db = bool(cfg.get("use_database", True))
        if self.use_db:
            self.engine = create_engine(
                f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}"
                f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            )

    # Ejecutar consulta SQL
    def ejecutar_consulta(self, query: str | None = None, sql_path: str | None = None):
        if not self.use_db:
            raise RuntimeError("La conexión a base de datos está desactivada en config.json")

        if query is None:
            base_dir = os.path.dirname(__file__)
            sql_file = sql_path or os.path.join(base_dir, "consulta.sql")
            with open(sql_file, "r") as f:
                query = f.read()

        df = pd.read_sql(text(query), self.engine)
        return df

    # Leer archivo plano
    def cargar_archivo(self, ruta: str, formato: str | None = None):
        """
        Lee un archivo local o desde DBFS.
        Ejemplo: df = lector.cargar_archivo("/dbfs/FileStore/mi_archivo.csv")
        """
        if formato is None:
            formato = os.path.splitext(ruta)[1].lstrip(".").lower()

        match formato:
            case "csv":
                return pd.read_csv(ruta)
            case "parquet":
                return pd.read_parquet(ruta)
            case "json":
                return pd.read_json(ruta, lines=True)
            case "xlsx" | "excel":
                return pd.read_excel(ruta)
            case _:
                raise ValueError(f"Formato '{formato}' no soportado.")

    # Exportar DataFrame
    def exportar(
        self,
        df: pd.DataFrame,
        nombre_archivo: str = "resultado",
        formato: str = "parquet",
        destino: str | None = None,
    ):
        if df is None or df.empty:
            print("El DataFrame está vacío, no se exporta.")
            return

        out_dir = destino or self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{nombre_archivo}.{formato}")

        try:
            match formato:
                case "parquet":
                    df.to_parquet(path)
                case "csv":
                    df.to_csv(path, index=False)
                case "json":
                    df.to_json(path, orient="records", lines=True)
                case "xlsx" | "excel":
                    df.to_excel(path, index=False)
                case _:
                    raise ValueError(f"Formato '{formato}' no soportado.")

            print(f"Archivo guardado en: {path}")
        except Exception as e:
            print(f"Error al exportar → {e}")