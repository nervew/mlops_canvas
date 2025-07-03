# Databricks notebook source
# MAGIC %md
# MAGIC # Exploración de Datos

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instalación de Liberias

# COMMAND ----------

# DBTITLE 1,Liberrias de Análisis Avanzado
# Instalar bamboolib
# https://docs.databricks.com/en/notebooks/bamboolib.html
%pip install bamboolib

# Instalar sweetviz
%pip install sweetviz

# Reiniciamos el python eviroment
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importar liberías

# COMMAND ----------

# DBTITLE 1,Librerias Basicas de Análisis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# COMMAND ----------

# DBTITLE 1,Librerías de Análisis Avanzadas
import bamboolib as bam
import sweetviz as sv
from IPython.display import display, HTML

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura de los datos

# COMMAND ----------

# paths
#/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/data/raw/query.sql
#https://adb-935665325582595.15.azuredatabricks.net/editor/files/4443600637143516?o=935665325582595
#data/raw/query.sql

#query =""
#df = spark.sql(query).toPandas()
parquet_path ="/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/data/raw/usuarios_unicos.parquet"
df = pd.read_parquet(parquet_path)
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis Varios

# COMMAND ----------

# DBTITLE 1,Mostrar primeras filas del dataframe
df.head()

# COMMAND ----------

# DBTITLE 1,Obtener Información General del DataFrame
df.info()

# COMMAND ----------

# DBTITLE 1,Describir Estadísticas Básicas
df.describe()

# COMMAND ----------

# Seleccionar columnas con tipo 'object'
numeric_cols = df[df.select_dtypes(exclude=['object']).columns]

# COMMAND ----------

# DBTITLE 1,Histograma de todas las variables numéricas
# Histograma de todas las variables numéricas
#numeric_cols.hist(bins=50, figsize=(20,15))
#plt.show()
# 1. Detectar automáticamente la columna datetime
datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns

# Validar que haya solo una columna datetime
if len(datetime_cols) != 1:
    raise ValueError("Debe haber exactamente una columna de tipo datetime.")

time_col = datetime_cols[0]  # Guardamos el nombre

# 2. Detectar columnas numéricas (sin incluir la datetime)
numeric_cols = df.select_dtypes(include=['number']).columns.drop(time_col, errors='ignore')

# 3. Graficar cada columna numérica vs tiempo
for col in numeric_cols:
    plt.figure(figsize=(12, 4))
    plt.plot(df[time_col], df[col], marker='o')
    plt.title(f'{col} a lo largo del tiempo')
    plt.xlabel(time_col)
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# DBTITLE 1,Mapa de calor de la correlación

numeric_df = df.select_dtypes(include='number')

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Mapa de calor de la correlación")
    plt.show()
else:
    print("No hay suficientes variables numéricas para hacer un mapa de calor.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis Librerias Especificas de Profiling

# COMMAND ----------

# Activar Bamboolib
bam.enable()

# Genera el reporte
df

# COMMAND ----------

# Definir la ruta donde se guardará el reporte
reporte_dir = "/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/reports/eda"
reporte_path = os.path.join(reporte_dir, "sweetviz_report.html")

# Crear la carpeta si no existe
os.makedirs(reporte_dir, exist_ok=True)

# Asegurar que la columna datetime esté en formato datetime64[ns]
df['mes_registro'] = pd.to_datetime(df['mes_registro'])

# Generar el análisis exploratorio
reporte = sv.analyze(df)

# Guardar el archivo HTML
reporte.show_html(reporte_path)

# Mostrarlo directamente en el notebook (opcional)
with open(reporte_path, "r", encoding="utf-8") as f:
    sweetviz_html = f.read()

display(HTML(sweetviz_html))
