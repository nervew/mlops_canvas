# feature_engineering

Módulo de Ingeniería de Variables para ML Pipeline  
Convierte datos crudos en una matriz de características lista para modelado,  
aplicando imputación, escalado, codificación y transformación de fechas.

---

## ¿Qué hace este módulo?

1. **Imputación**  
   - Rellena valores faltantes en columnas numéricas (mediana) y categóricas (valor más frecuente).

2. **Escalado**  
   - Todas las variables numéricas y derivadas (como año, mes, día) se convierten para tener media 0 y varianza 1 (`StandardScaler`).

3. **Codificación One-Hot**  
   - Variables categóricas se convierten a columnas binarias, una por cada valor posible (incluyendo los valores nulos si los hay).

4. **Procesamiento de Fechas**  
   - Convierte columnas de fecha a tres columnas: año, mes, día, y luego las escala.

---

## Carpetas
```
feature_engineering/
│
├── application/
│ └── service.py
│ # Función principal:
│ # run(df: pd.DataFrame, target: str = "target") -> FeatureTransformer
│ # - Recibe:
│ # - df: DataFrame original (con variables numéricas, categóricas, fechas y columna objetivo)
│ # - target: nombre de la columna objetivo (por defecto "target")
│ # - Retorna:
│ # - FeatureTransformer (objeto de dominio con el pipeline y metadatos)
│
├── domain/
│ └── transformer.py
│ # Clase principal:
│ # FeatureTransformer
│ # - Recibe:
│ # - transformer: pipeline de scikit-learn con todas las transformaciones aplicadas
│ # - created_at: timestamp de creación
│ # - version: versión del pipeline
│ # - Métodos:
│ # - fit(X, y=None): ajusta el pipeline a los datos
│ # - transform(X): transforma nuevos datos
│ # - fit_transform(X, y=None): ajusta y transforma de una vez
│ # - get_output_feature_names(): retorna lista de nombres de columnas generadas
│ # - save(path): guarda el pipeline y metadatos
│ # - load(path): carga el pipeline guardado
│
├── infrastructure/
│ ├── robust_fe.py
│ │ # Función principal:
│ │ # build_transformer(df: pd.DataFrame, target: str = "target") -> Pipeline
│ │ # - Recibe:
│ │ # - df: DataFrame original (sin procesar)
│ │ # - target: nombre de la columna objetivo
│ │ # - Retorna:
│ │ # - Pipeline de scikit-learn (con todas las etapas de ingeniería de variables)
│ │ #
│ │ # get_feature_names(preprocessor: ColumnTransformer) -> list[str]
│ │ # - Recibe:
│ │ # - preprocessor: ColumnTransformer del pipeline
│ │ # - Retorna:
│ │ # - Lista de nombres de columnas generadas por el preprocesamiento
│ │
│ └── custom_transformers.py
│ # Clase principal:
│ # DateDecomposer
│ # - Recibe:
│ # - DataFrame con columnas datetime
│ # - Retorna:
│ # - DataFrame con columnas año, mes y día derivadas de cada columna datetime
│
└── init.py
# Expone los puertos públicos:
# from .application.service import run
# from .domain.transformer import FeatureTransformer
```
## Ejemplo de uso

```python
import pandas as pd
from feature_engineering.application.service import run as fe_run

df = pd.DataFrame({
    "edad": [25, 32, None, 47],
    "ingresos": [50000, 60000, 40000, None],
    "genero": ["M", "F", None, "F"],
    "fecha_registro": pd.to_datetime([
        "2023-01-01", "2023-06-15", "2023-02-10", "2022-12-25"
    ]),
    "target": [0, 1, 0, 1],
})

ft = fe_run(df, target="target")
X = ft.fit_transform(df)
cols = ft.get_output_feature_names()
pd.DataFrame(X, columns=cols).round(2)
