#!/usr/bin/env bash
set -euo pipefail

# Sube modelo, requirements y esquema a Azure Blob Storage siguiendo la estructura solicitada.

if [[ $# -lt 1 ]]; then
  echo "Uso: MODEL_NAME=<nombre> STORAGE_ACCOUNT=<cuenta> ./scripts/upload_assets.sh <ruta_local_base>" >&2
  echo "Ejemplo: ./scripts/upload_assets.sh artifacts/my_model" >&2
  exit 1
fi

LOCAL_BASE="$1"
MODEL_NAME=${MODEL_NAME:-$(basename "$LOCAL_BASE")}
RESOURCE_GROUP=${RESOURCE_GROUP:-GRPANALITICA}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:?"Defina STORAGE_ACCOUNT (prefijo mlops)"}
MODEL_CONTAINER=${MODEL_CONTAINER:-models}

if [[ ! -d "$LOCAL_BASE" ]]; then
  echo "La ruta local $LOCAL_BASE no existe" >&2
  exit 1
fi

az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination "$MODEL_CONTAINER" \
  --destination-path "$MODEL_NAME/model" \
  --source "$LOCAL_BASE/model" \
  --auth-mode login \
  --overwrite

az storage blob upload \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$MODEL_CONTAINER" \
  --name "$MODEL_NAME/requirements/requirements.txt" \
  --file "$LOCAL_BASE/requirements/requirements.txt" \
  --auth-mode login \
  --overwrite

az storage blob upload-batch \
  --account-name "$STORAGE_ACCOUNT" \
  --destination "$MODEL_CONTAINER" \
  --destination-path "$MODEL_NAME/data" \
  --source "$LOCAL_BASE/data" \
  --auth-mode login \
  --overwrite

echo "Carga completada para $MODEL_NAME en contenedor $MODEL_CONTAINER de $STORAGE_ACCOUNT"
