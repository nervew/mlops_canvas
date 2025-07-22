# Módulo eda_univariado

Realiza el análisis exploratorio univariado de cualquier DataFrame de entrada.
Produce un reporte exhaustivo en JSON con estadísticas descriptivas, análisis de valores nulos, métricas de outliers y datos de histogramas para cada variable.

Adicionalmente, imprime solo las primeras 5 filas del resumen estándar de pandas.DataFrame.describe().T para inspección rápida.
```
eda_univariado/
│
├── application/
│   └── service.py          
│       # Orquesta el análisis univariado
│       # Recibe: DataFrame, ruta de salida (opcional)
│       # Imprime: describe().T.head() (primeras 5 filas)
│       # Llama a infraestructura para análisis y guardado
│       # Salida: UnivariateReport (dict serializable)
│
├── domain/
│   └── report.py           
│       # Entidad UnivariateReport: solo estructura de datos
│       # No depende de pandas ni de lógica de análisis
│
└── infrastructure/
    └── pandas_univar.py    
        # Analiza el DataFrame y genera métricas extendidas
        # Convierte todos los tipos a nativos de Python
        # Asegura que la carpeta de salida existe:
        #   - Carpeta de salida por defecto: ./output/eda_report/
        #   - Archivo generado: eda_report_univariado.json
        # Guarda el reporte extendido en JSON
```
# Ejemplo de uso
```
import pandas as pd
from app.eda_univariado.application import service as eda_uni

df = pd.DataFrame({
    "edad": [22, 30, 35, 40, 22, 35, 35, 1000],
    "ingresos": [1500, 2000, 1800, 2200, 1600, 2100, 1800, 1900],
    "sexo": ["M", "F", "F", "M", "F", "M", None, "F"],
    "hijos": [0, 1, 2, 1, 0, 0, 1, 2],
})

output_path = "./output/eda_report/eda_report_univariado.json"
eda_report = eda_uni.run(df, output_path)
# Se imprime por consola: las primeras 5 filas de describe().T
# El reporte detallado queda guardado en output_path (JSON)

```



