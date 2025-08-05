# Feature Engineering Module

Módulo de Ingeniería de Variables para ML Pipeline  
Convierte datos crudos en una matriz de características lista para modelado,  
aplicando imputación, escalado, codificación y transformación de fechas.

---

## ¿Qué hace este módulo?

1. **Imputación**  
   - Rellena valores faltantes en columnas numéricas (mediana)

2. **Escalado**  
   - Todas las variables numéricas se convierten para tener media 0 y varianza 1 (`StandardScaler`).

3. **Codificación One-Hot**  
   - Variables categóricas se convierten a columnas binarias, una por cada valor posible (incluyendo los valores nulos si los hay).

4. **Procesamiento de Fechas**  
   - Convierte columnas de fecha a tres columnas: año, mes, día, y luego las escala.

---

## Carpetas

```
m05_feature_engineering/
├── __init__.py
├── pipeline_engineering.py
├── readme_feature_en.md
├── step01_import/
│   ├── adapters/
│   └── ports/
├── step02_imputation/
│   ├── adapters/
│   └── ports/
├── step03_outliers/
│   ├── adapters/
│   └── ports/
├── step04_transformation/
│   ├── adapters/
│   └── ports/
├── step05_encoding/
│   ├── adapters/
│   └── ports/
├── step06_feature_gen/
│   ├── adapters/
│   └── ports/
├── step07_metrics/
│   ├── adapters/
│   └── ports/
├── step08_drift/
│   ├── adapters/
│   └── ports/
├── step09_export/
│   ├── adapters/
```

## Imputación

Este módulo soporta la imputación de valores faltantes tanto para variables numéricas como categóricas:

- **Columnas numéricas:** Imputadas usando la mediana.
- **Columnas categóricas:** Imputadas usando el valor más frecuente (moda).

### Ejemplo

```python
from step02_imputation.adapters.simple_imputer import SimpleImputerAdapter

imputer = SimpleImputerAdapter()
df_imputed = imputer.fit_transform(df)
```

## Exportación de Métricas

Después de la ingeniería de características, se exportan dos archivos en la carpeta `metrics`:

### `feature_metrics.json`

Contiene:
- Número de filas y columnas
- Medias de todas las columnas numéricas

### `drift.json`

Contiene:
- Un diccionario con el PSI (Population Stability Index) para cada columna numérica comparando los conjuntos de train y test.

#### Definición de PSI

- **PSI < 0.1** → Sin cambio significativo, la variable es estable.
- **0.1 ≤ PSI < 0.25** → Cambio moderado, monitorea esta variable.
- **PSI ≥ 0.25** → Cambio significativo, posible drift.

### Ejemplo de Uso

```python
from step01_import.adapters.parquet_loader import ParquetPartitionLoader
from step02_imputation.adapters.simple_imputer import SimpleImputerAdapter

# Cargar datos
loader = ParquetPartitionLoader("/ruta/a/datos/crudos/particionados")
train, test, back = loader.load()

# Imputar valores faltantes
imputer = SimpleImputerAdapter()
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# Cálculo de métricas y drift (simplificado)
import numpy as np
import json

metrics = {
    "n_rows": train_imputed.shape[0],
    "n_cols": train_imputed.shape[1],
    "means": train_imputed.mean(numeric_only=True).to_dict()
}

# Ejemplo de cálculo de PSI para una columna
def calculate_psi(train_col, test_col, bins=10):
    train_bins = np.histogram(train_col, bins=bins)[0] / len(train_col)
    test_bins = np.histogram(test_col, bins=bins)[0] / len(test_col)
    psi = np.sum((train_bins - test_bins) * np.log((train_bins + 1e-6) / (test_bins + 1e-6)))
    return psi

drift = {col: calculate_psi(train_imputed[col], test_imputed[col])
         for col in train_imputed.select_dtypes(include="number").columns}

# Exportar métricas
with open("metrics/feature_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("metrics/drift.json", "w") as f:
    json.dump(drift, f, indent=2)
```

## Pasos

- step01_import: Carga de datos
- step02_imputation: Imputación de valores faltantes
- step03_outliers: Detección y manejo de outliers
- step04_transformation: Transformación de datos
- step05_encoding: Codificación categórica
- step06_feature_gen: Generación de características
- step07_metrics: Cálculo de métricas
- step08_drift: Detección de drift en los datos
- step09_export: Exportación de datos procesados

## Uso

Cada paso es modular y puede ser utilizado independientemente o como parte del flujo de trabajo completo.