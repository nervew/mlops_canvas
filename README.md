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

## Despliegue de la API de prueba en Azure Container Apps

El repositorio incluye una API mínima (`api/main.py`) que responde "hola mundo" y scripts automatizados para construir la imagen, publicarla en Azure Container Registry (ACR) y desplegarla en Azure Container Apps. Todos los recursos usan el prefijo `mlopstest` y se crean en el grupo de recursos `GRPANALITICA` dentro de la suscripción `Gobierno de datos`.

1. **Autenticación y preparación local**
   ```bash
   az login
   az account set --subscription "Gobierno de datos"
   ```

2. **Construir y publicar la imagen en ACR**
   ```bash
   python scripts/build_and_push.py --image-version v1
   ```
   - Usa el `Dockerfile` en la raíz para empaquetar la API FastAPI.
   - Etiqueta la imagen como `mlopstestacr.azurecr.io/mlopstest-api:v1` y la publica en ACR.

3. **Provisionar los recursos de Azure (ACR, Log Analytics, Container Apps Environment y Container App)**
   ```bash
   ./scripts/provision_resources.sh
   ```
   - Utiliza la plantilla Bicep `infra/provision.bicep` para crear o actualizar los recursos requeridos.

4. **Actualizar el contenedor desplegado con una nueva imagen**
   ```bash
   ./scripts/deploy_container_app.sh
   ```
   - Apunta la Container App `mlopstest-api` a la imagen publicada en ACR.
   - Para consultar la URL pública: `az containerapp show --name mlopstest-api --resource-group GRPANALITICA --query properties.configuration.ingress.fqdn -o tsv`.

Variables de entorno opcionales para personalizar: `SUBSCRIPTION`, `RESOURCE_GROUP`, `LOCATION`, `PREFIX`, `CONTAINER_IMAGE` e `IMAGE`.

---

## Contribuciones

Este repositorio está diseñado para trabajo colaborativo con enfoque DevOps/MLOps.  
Se recomienda usar ramas feature, PRs para integración y herramientas CI/CD para despliegue automático.

---

## Contacto

Para dudas o colaboración, contactar con el equipo de MLOps.

---
