# mlops_canvas
Algotihms for create a full pipeline MLOps



## Ejecución Rápida

Instala las dependencias automáticamente al ejecutar el pipeline en /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src

python -m app.pipeline

El pipeline instalará automáticamente los paquetes listados en /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/requirements.txt

## Estructura de carpetas 
```plaintext
src/app/
    data_ingestion/
    eda_univariado/
    data_validation/
    data_split/
    search_model/
    pipeline.py

models/
    model_v1.joblib
    model_v1.onnx

output/
    reports/eda_univariado.html
    validation/data_profile.json
    validation/report.json
```
## Resumen del flujo

| Paso/Módulo        | Recibe           | Entrega                    | Ruta de salida principal            |
| ------------------ | ---------------- | -------------------------- | ----------------------------------- |
| Ingesta            | config/dataset   | DataFrame                  | -                                   |
| EDA Univariado     | DataFrame        | Estadísticos + HTML        | output/reports/eda\_univariado.html |
| Validación         | DataFrame        | Perfil + Validación        | output/validation/\*                |
| Split (train/test) | DataFrame        | DFs: train, test, backtest | -                                   |
| AutoML FLAML       | DFs split        | Modelos, hiperparámetros   | -                                   |
| Guardado de modelo | Modelo entrenado | .joblib, .onnx             | models/       


## Flujo de Módulos, Entradas y Salidas

1. Ingesta de datos

Módulo: app.data_ingestion.application.service.ingest

Entrada: Parámetros internos (ruta/dataset predefinido)

Salida: Objeto con el DataFrame (dataset.data)

2. Análisis Univariado (EDA)
Módulo: app.eda_univariado.application.service.run

Entrada: df (DataFrame)

Salida: Objeto UnivariateReport con:

description: Estadísticos básicos (pandas.DataFrame)

html_report_path: Ruta del HTML generado (por el momento pausado futuras entregas)

Ruta de almacenamiento:

Reporte HTML: output/reports/eda_univariado.html


3. Validación de datos
Módulo: app.data_validation.application.service.run

Entrada: df (DataFrame), fit_profile=True/False

Salida: Objeto ValidationReport con:

valido (bool): Si pasa la validación

detalles (dict): Problemas detectados

Ruta de almacenamiento:

Perfil: output/validation/data_profile.json

Reporte de validación: output/validation/report.json


4. Split robusto (entrenamiento, test, backtest)
Módulo: app.data_split.application.service.run

Entrada: df (DataFrame), parámetros de split

Salida: Objeto con:

train_df (DataFrame)

test_df (DataFrame)

backtest_df (DataFrame)

Ruta de almacenamiento:

No guarda automáticamente en disco (usa en memoria).

5. Entrenamiento y búsqueda automática de modelo
Módulo: app.search_model.flaml_wrapper.FLAMLWrapper

Entrada: X_train, y_train, X_test, y_test

Salida:

model_ranking (DataFrame): Ranking de modelos probados

best_model: (nombre, objeto modelo)

best_params (dict): Hiperparámetros óptimos

Ruta de almacenamiento:

No guarda modelos aquí (solo en memoria para siguiente paso).


6. Guardado del modelo
Función: save_model

Entrada: Mejor modelo (best_model_obj), X_train

Salida:

Archivo .joblib con el modelo entrenado (siempre)

Archivo .onnx (si el modelo es compatible con sklearn-onnx)

Ruta de almacenamiento:

Joblib: models/model_v1.joblib

ONNX: models/model_v1.onnx (si es compatible)

                      |


Ruta de almacenamiento: No guarda archivos (trabaja en memoria).