#!/usr/bin/env bash
set -euo pipefail

: "${STORAGE_ACCOUNT:?Debe definir STORAGE_ACCOUNT}"
: "${MODEL_NAME:?Debe definir MODEL_NAME}"
: "${ACR_NAME:?Debe definir ACR_NAME}"
: "${IMAGE_TAG:=v1}"  # default if not provided

MODEL_CONTAINER=${MODEL_CONTAINER:-models}
BLOB_REQUIREMENTS_PATH=${BLOB_REQUIREMENTS_PATH:-"${MODEL_NAME}/requirements/requirements.txt"}
RUNTIME_REQ_FILE="infra/azure_inference_service/runtime_requirements.txt"
BASE_REQ_FILE="infra/azure_inference_service/base.requirements.txt"
TMP_REQ=$(mktemp)
IMAGE_NAME="${ACR_NAME}.azurecr.io/${MODEL_NAME}-inference:${IMAGE_TAG}"

if ! command -v az >/dev/null; then
  echo "Azure CLI es requerido" >&2
  exit 1
fi

# Descarga requirements del modelo y combina con las bases del servicio
az storage blob download \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$MODEL_CONTAINER" \
  --name "$BLOB_REQUIREMENTS_PATH" \
  --file "$TMP_REQ" \
  --auth-mode login

cat "$BASE_REQ_FILE" "$TMP_REQ" | sort -u > "$RUNTIME_REQ_FILE"
echo "Requisitos combinados escritos en $RUNTIME_REQ_FILE"

# Construye y publica imagen
az acr login --name "$ACR_NAME"
docker build -t "$IMAGE_NAME" -f infra/azure_inference_service/Dockerfile .
docker push "$IMAGE_NAME"

echo "Imagen publicada: $IMAGE_NAME"
