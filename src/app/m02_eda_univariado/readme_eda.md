# Módulo m01_eda_univariado

Este módulo realiza el Análisis Exploratorio de Datos (EDA) **univariado** sobre un `DataFrame` de Pandas, calculando estadísticas descriptivas “a mano” y generando un reporte interactivo en HTML con **Sweetviz**.

Adicionalmente, imprime solo las primeras 5 filas del resumen estándar de pandas.DataFrame.describe().T para inspección rápida.
```
    ├── m00_instalador/             # Instalación de dependencias
    │   ├── __init__.py
    │   └── service.py
    ├── m01_eda_univariado/         # EDA univariado con Sweetviz
    │   ├── application/
    │   │   └── run.py
    │   ├── domain/
    │   │   └── report.py
    │   └── infrastructure/
    │       └── pandas_univar.py

output/
└── reporte_eda/                     # Reportes HTML generados por Sweetviz
```
# Ejemplo de uso
```
import numpy as np
import pandas as pd
from app.m01_eda_univariado import run as run_eda

# Dataset de prueba
df = pd.DataFrame({
    "edad": np.random.randint(18, 70, size=50),
    "sexo": np.random.choice(["M","F"], size=50),
})

# Ejecutar módulo
report = run_eda(df)

# Ver estadísticas en consola
# -> df.describe().T.head()

# HTML en:
# output/reporte_eda/sweetviz_univariate_report.html

print(report.description["edad"]["mean"])  # Ejemplo de acceso


```



