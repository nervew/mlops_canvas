#!/usr/bin/env bash
set -euo pipefail

# Descarga requirements del modelo desde BlobStorage, construye la imagen y la publica en ACR.

MODEL_NAME=${MODEL_NAME:?"Defina MODEL_NAME"}
RESOURCE_GROUP=${RESOURCE_GROUP:-GRPANALITICA}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:?"Defina STORAGE_ACCOUNT"}
MODEL_CONTAINER=${MODEL_CONTAINER:-models}
ACR_NAME=${ACR_NAME:?"Defina ACR_NAME"}
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE_NAME=${IMAGE_NAME:-${ACR_NAME}.azurecr.io/${MODEL_NAME}:${IMAGE_TAG}}
BUILD_DIR=${BUILD_DIR:-build}

mkdir -p "$BUILD_DIR"

az storage blob download \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$MODEL_CONTAINER" \
  --name "$MODEL_NAME/requirements/requirements.txt" \
  --file "$BUILD_DIR/requirements.model.txt" \
  --auth-mode login \
  --overwrite

cat src/inference_api/base-requirements.txt "$BUILD_DIR/requirements.model.txt" > "$BUILD_DIR/requirements.lock"

docker build --file docker/inference.Dockerfile --tag "$IMAGE_NAME" .
az acr login --name "$ACR_NAME"
docker push "$IMAGE_NAME"

echo "Imagen publicada: $IMAGE_NAME"
