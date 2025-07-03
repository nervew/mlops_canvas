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
# MLOps Pipeline - Proyecto Ejemplo

## Resumen del flujo

Este pipeline automatiza el proceso de:

1. **Ingesta de datos**
2. **Análisis exploratorio univariado (EDA)**
3. **Validación robusta de datos**
4. **División (split) en train/test/backtest**
5. **Búsqueda automática y entrenamiento del mejor modelo**
6. **Guardado del modelo (joblib y ONNX, si es posible)**

---

## Estructura del pipeline

Ingesta ▶ EDA ▶ Validación ▶ Split ▶ AutoML ▶ Guardado de modelo


---

## 📦 Qué hace cada módulo

| Módulo                          | ¿Qué hace?                                                  | ¿Qué recibe?           | ¿Qué retorna?                         |
|----------------------------------|-------------------------------------------------------------|------------------------|---------------------------------------|
| `data_ingestion`                 | Carga datos de la fuente (CSV, base, etc.)                  | (opcional: config)     | Objeto `Dataset` con `.data` (DataFrame) |
| `eda_univariado`                 | Analiza estadística y distribuciones de cada variable       | DataFrame              | Reporte con `.description` y/o `.html_report_path` |
| `data_validation`                | Valida estructura, tipos, outliers, reglas aprendidas       | DataFrame, fit_profile | Objeto con `.valido` (bool), `.detalles` (dict)    |
| `data_split`                     | Divide datos en train, test y backtest                      | DataFrame, params de split | Objeto con `.train_df`, `.test_df`, `.backtest_df` |
| `search_model` (`flaml_wrapper`) | Busca el mejor modelo de forma automática (AutoML)          | X_train, y_train, X_test, y_test | Mejor modelo, ranking, hiperparámetros |
| `save_model` (función)           | Guarda el modelo entrenado en joblib y ONNX (si es compatible) | Modelo, X_train        | Archivos `.joblib` y `.onnx` (si aplica) |

---

## ▶️ Ejecución rápida

1. **Instala dependencias:**
   ```bash
   pip install -r requirements.txt

2. para correr el pipeline desde src/

   python -m app.pipeline

