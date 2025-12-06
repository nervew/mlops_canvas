# Pipeline de inferencia en Azure (Blob Storage + ACR + Container Apps)

Esta carpeta contiene el servicio de inferencia (FastAPI) y scripts para construir y desplegar imágenes en entornos **dev** y **staging**.

## Flujo de alto nivel
1. **Publicar artefactos** en Azure Blob Storage bajo `model_name/`:
   - `model/<model_name>.pickle` o `.onnx`
   - `requirements/requirements.txt`
   - `data/schema.empty.parquet` (opcional)
2. **Construir imagen** de inferencia combinando `requirements.txt` del modelo con dependencias base.
3. **Publicar imagen** en Azure Container Registry (ACR).
4. **Desplegar** la imagen en Azure Container Apps para dev y staging.

## Variables comunes
- `MODEL_NAME`: nombre del modelo y prefijo en Blob Storage.
- `MODEL_CONTAINER`: contenedor en Blob Storage (por defecto `models`).
- `STORAGE_ACCOUNT`: nombre de la cuenta de almacenamiento.
- `ACR_NAME`: nombre de ACR sin sufijo `azurecr.io`.
- `RESOURCE_GROUP`, `ACA_ENVIRONMENT`: recursos de destino para Container Apps.
- `AZURE_STORAGE_CONNECTION_STRING`: conexión para lectura desde Container Apps.

## Scripts
- `scripts/upload_model_artifacts.sh`: sube modelo, requirements y esquema usando `az storage blob upload`.
- `scripts/build_and_push_image.sh`: descarga `requirements.txt` desde Blob Storage, lo combina con `infra/azure_inference_service/base.requirements.txt`, genera `runtime_requirements.txt`, construye la imagen y la publica en ACR.
- `scripts/deploy_container_app.sh`: despliega la imagen en dos Container Apps (`-dev` y `-stg`) con variables de entorno para conectar al Blob Storage.

## Servicio FastAPI
- Endpoint `GET /health`: verificación básica.
- Endpoint `POST /infer`:
  - Datos de entrada como JSON (`data`) o archivo (`csv/parquet/json`).
  - Parámetros opcionales `model_name` y `model_blob_path` para elegir versión del modelo.
  - Descarga el modelo desde Blob Storage (pickle u ONNX) y ejecuta predicciones.
  - Retorna JSON con `predictions` y `prediction_count`.

## Ejecución local
```bash
export AZURE_STORAGE_CONNECTION_STRING="..."
export MODEL_BLOB_CONTAINER=models
uvicorn infra.azure_inference_service.main:app --reload --port 8080
```

El archivo `infra/azure_inference_service/runtime_requirements.txt` se genera automáticamente, pero se incluye una versión base para permitir builds locales.
