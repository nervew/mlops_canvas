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


query =""
df = spark.sql( query).toPandas()

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
numeric_cols.hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

# DBTITLE 1,Mapa de calor de la correlación
# Mapa de calor de la correlación
plt.figure(figsize=(12,10))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análisis Liberias Especificas de Profiling

# COMMAND ----------

# Activar Bamboolib
bam.enable()

# Genera el reporte
df

# COMMAND ----------

# Generar el reporte
report = sv.analyze(df)

# Guardar el reporte como HTML
report.show_html("EDA/sweetviz_report.html")

# Mostrar el reporte directamente en el notebook (si es compatible)
with open("EDA/sweetviz_report.html", "r") as f:
    sweetviz_html = f.read()
display(HTML(sweetviz_html))
