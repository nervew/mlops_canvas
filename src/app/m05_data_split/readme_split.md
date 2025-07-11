# Módulo data_split
Permite dividir un DataFrame en subconjuntos de entrenamiento, test y backtest de manera controlada y reproducible. Soporta varios métodos de split y el cálculo de métricas de estabilidad entre conjuntos, para validar que la muestra de entrenamiento y test sean comparables y evitar data leakage o drift.
```
data_split/
│
├── application/
│   └── service.py          # Orquesta el proceso de split, valida parámetros de entrada
│
├── domain/
│   └── result.py           # Define la estructura DataSplitResult (train, test, backtest)
│
└── infrastructure/
    └── robust_data_splitter.py # Lógica completa del splitting, métricas y visualización

```
## Métodos soportados
Random split:

División aleatoria y estratificada (target + columnas adicionales).

Time split:

División cronológica usando una columna de fechas.

Metric split:

División por cuantiles de una columna numérica.

# Ejemplo de uso
import pandas as pd
from data_split.application.service import run

# DataFrame de ejemplo
df = pd.read_csv("datos.csv")

split_result = run(
    df,
    split_method="random",
    target_column="target",
    stratify_columns=["sexo"],
    train_size=0.7,
    test_size=0.2,
    backtest_size=0.1
)

train_df = split_result.train_df
test_df = split_result.test_df
backtest_df = split_result.backtest_df

# Ejemplo: calcular métricas de estabilidad
from data_split.infrastructure.robust_data_splitter import RobustDataSplitter
splitter = RobustDataSplitter(
    df,
    split_method="random",
    target_column="target",
    stratify_columns=["sexo"],
    train_size=0.7,
    test_size=0.2,
    backtest_size=0.1
)
splitter.split_data()
metrics = splitter.calculate_metrics()
print(metrics)




