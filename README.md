# Proyecto Pipeline MLOps - Entrenamiento y Producción de Modelos

Este proyecto implementa un pipeline modular y escalable para el ciclo completo de Machine Learning, desde la ingestión de datos hasta la validación post-despliegue, siguiendo buenas prácticas de MLOps, reproducibilidad, monitoreo y trazabilidad.

---

## Arquitectura General

El pipeline está dividido en pasos claros y desacoplados, cada uno implementado con clases especializadas para facilitar mantenimiento, escalabilidad y trazabilidad.

Las etapas principales son:

| Paso                          | Entrada                                   | Proceso clave                                 | Salida                                |
|-------------------------------|-------------------------------------------|-----------------------------------------------|-------------------------------------|
| **Data Ingestion**             | Fuentes externas (archivos, bases de datos) | Lectura y validación de datos, versionado     | `data_raw (pandas.DataFrame)`       |
| **EDA Univariado**             | `data_raw`                                | Estadísticas descriptivas, detección de outliers | `eda_report_univariado (dict/json)` |
| **Validación de Data Ingestion** | `data_raw`, `target`                      | Validación de tipos, nulos, duplicados, reglas | `validacion_data_ingestion (dict/bool)` |
| **Validación de Inferencias**  | `data_raw`, `target`                      | Validación de columna de inferencia           | `validacion_inferencia (dict/bool)` |
| **Feature Engineering**        | `data_raw`                               | Creación y transformación de variables        | `transformador_inicial`, `data_transformada` |
| **Feature Selection**          | `data_transformada`, `target`            | Selección de variables relevantes              | `transformador_feature_selection`, `data_features_seleccionadas` |
| **Split DataSet**              | `data_features_seleccionadas`, `target` | División en train, test y validación           | `X_train, X_test, X_val`, `y_train, y_test, y_val` |
| **Search Model**               | `X_train, y_train, X_test, y_test`       | Entrenamiento y evaluación de modelos candidatos | `modelo_mejor`, `metricas_modelo`  |
| **Search Hiperparams**         | `X_train, y_train, modelo_mejor`          | Optimización de hiperparámetros                 | `mejores_hiperparametros`, `modelo_optimizado` |
| **Feature Reduction**          | `transformador_inicial`, lista de features | Reducción de dimensionalidad                    | `transformador_final`               |
| **Create Model**               | `X_train, y_train, modelo_optimizado, transformador_final` | Entrenamiento final y serialización ONNX       | `pipeline_serializado_onnx`, `modelo_final` |
| **Inferencia**                 | `pipeline_serializado_onnx`, `datos_nuevos` | Aplicación de pipeline para nuevas predicciones | `predicciones`                     |
| **Validación de Inferencias**  | `predicciones`, `reglas_inferencia`       | Validación de predicciones                      | `validacion_inferencia_post`       |
| **Validate Model**             | `modelo_final`, `modelo_viejo`, `X_val`, `y_val` | Comparación de modelos y detección de drift    | `resultado_validacion`, `alerta_drift`, `decision_final` |
| **EDA Bivariado**              | `modelo_final`, `X_val`, `y_val`          | Interpretación e importancia de variables      | `eda_bivariado_report`              |
| **EDA Scores**                 | `predicciones`, `y_val`                    | Análisis de métricas y performance              | `reportes_metricas`                 |

---

## Arquitectura de Clases

El pipeline está implementado con clases modulares que encapsulan cada etapa, facilitando pruebas, trazabilidad y monitoreo continuo.  
Ejemplo de clases clave:

- `DataIngestion`: Lectura y versionado de datos.
- `EDAUnivariado`: Reportes estadísticos univariados.
- `DataValidation` y `InferenceValidation`: Validaciones de calidad y consistencia.
- `FeatureEngineering` y `FeatureSelection`: Transformaciones y selección de variables.
- `DataSplitter`: División de datasets.
- `ModelSearcher` y `HyperparamSearcher`: Entrenamiento y optimización de modelos.
- `ModelTrainer`: Entrenamiento final y serialización.
- `InferenceRunner`: Ejecución de inferencias.
- `ModelValidator`: Validación post-despliegue y detección de drift.
- `EDABivariado` y `EDAScores`: Análisis avanzado e interpretación.

---

## Herramientas y Tecnologías

- Python 3.8+
- Pandas, NumPy para manipulación de datos
- Scikit-learn, XGBoost para modelos y transformadores
- ONNX para serialización de pipelines
- Great Expectations para validación de datos
- MLflow y DataBricks para trazabilidad, logging y orquestación
- Alertas integradas para monitoreo de desviaciones

---

## Uso y Ejecución

1. **Data Ingestion**  
   Cargar datos desde fuentes externas y versionar ingesta.

2. **Exploración y Validación**  
   Ejecutar análisis univariado y validar la calidad de datos.

3. **Feature Engineering y Selección**  
   Aplicar transformaciones y seleccionar variables relevantes.

4. **Entrenamiento y Optimización**  
   Entrenar modelos candidatos, buscar hiperparámetros óptimos.

5. **Entrenamiento Final y Serialización**  
   Generar pipeline final serializado en ONNX.

6. **Inferencia y Validación Post-Despliegue**  
   Realizar predicciones y validar la calidad de inferencias.

7. **Monitoreo Continuo y Reentrenamiento**  
   Validar modelo en producción, detectar drift y decidir reentrenamiento.

---

## Contribuciones

Este repositorio está diseñado para trabajo colaborativo con enfoque DevOps/MLOps.  
Se recomienda usar ramas feature, PRs para integración y herramientas CI/CD para despliegue automático.

---

## Contacto

Para dudas o colaboración, contactar con el equipo de MLOps.

---

## Pipeline de inferencia en Azure (dev y staging)

El pipeline de inferencia descarga dependencias desde Blob Storage, construye una imagen de inferencia y despliega dos entornos (dev y staging) en Azure Container Apps.

1. **Provisionar recursos**: `./scripts/install_resources.sh` crea (si no existen) el resource group `GRPANALITICA`, un Storage Account, ACR y un Container Apps Environment con prefijo `mlops`.
2. **Entrenar y exportar**: usa `notebooks/flaml_forecasting.ipynb` para entrenar con FLAML y generar artefactos en `artifacts/<model_name>/` (modelo `.pickle`, `requirements.txt` y esquema vacío en `data/`).
3. **Subir artefactos**: `MODEL_NAME=<model_name> STORAGE_ACCOUNT=<cuenta> ./scripts/upload_assets.sh artifacts/<model_name>` respeta la estructura `model_name/model|requirements|data`.
4. **Construir y publicar**: `MODEL_NAME=<model_name> STORAGE_ACCOUNT=<cuenta> ACR_NAME=<acr> ./scripts/build_and_push.sh` descarga `requirements.txt` desde Blob, combina con `src/inference_api/base-requirements.txt`, construye la imagen con `docker/inference.Dockerfile` y la envía a ACR.
5. **Desplegar dev y staging**: `ACR_NAME=<acr> IMAGE_NAME=<acr>.azurecr.io/<model>:<tag> STORAGE_ACCOUNT=<cuenta> ./scripts/deploy_container_app.sh` crea dos Container Apps (`mlops-dev-api` y `mlops-staging-api`) con variables de entorno para Blob Storage.

### API de inferencia
- Framework: FastAPI (`src/inference_api/app.py`).
- Endpoint `/infer`: acepta JSON (`records`) o archivo (`csv`, `parquet`, `pickle`), parámetro `model_name` y ruta opcional `model_blob_path`. Descarga modelos `.pickle` u `.onnx` desde Blob Storage, ejecuta `predict` o `onnxruntime` y retorna predicciones.
- Endpoint `/health`: verificación básica para Container Apps.
