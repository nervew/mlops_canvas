#!/usr/bin/env bash
set -euo pipefail

: "${RESOURCE_GROUP:?Debe definir RESOURCE_GROUP}"
: "${ACA_ENVIRONMENT:?Debe definir ACA_ENVIRONMENT}"
: "${ACR_NAME:?Debe definir ACR_NAME}"
: "${IMAGE_TAG:?Debe definir IMAGE_TAG}"
: "${MODEL_NAME:?Debe definir MODEL_NAME}"
: "${AZURE_STORAGE_CONNECTION_STRING:?Debe definir AZURE_STORAGE_CONNECTION_STRING}"

MODEL_CONTAINER=${MODEL_CONTAINER:-models}
LOCATION=${LOCATION:-"eastus"}
IMAGE="${ACR_NAME}.azurecr.io/${MODEL_NAME}-inference:${IMAGE_TAG}"
CONTAINER_APP_NAME_DEV=${CONTAINER_APP_NAME_DEV:-"${MODEL_NAME}-dev"}
CONTAINER_APP_NAME_STG=${CONTAINER_APP_NAME_STG:-"${MODEL_NAME}-stg"}

az acr login --name "$ACR_NAME"

for ENV_NAME in dev stg; do
  if [[ "$ENV_NAME" == "dev" ]]; then
    APP_NAME="$CONTAINER_APP_NAME_DEV"
  else
    APP_NAME="$CONTAINER_APP_NAME_STG"
  fi

  echo "\nDesplegando ${APP_NAME}..."
  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ACA_ENVIRONMENT" \
    --image "$IMAGE" \
    --target-port 8080 \
    --ingress external \
    --registry-server "${ACR_NAME}.azurecr.io" \
    --env-vars MODEL_NAME="$MODEL_NAME" MODEL_BLOB_CONTAINER="$MODEL_CONTAINER" LOG_LEVEL="INFO" \
    --secrets storage-conn="$AZURE_STORAGE_CONNECTION_STRING" \
    --env-vars AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn \
    --min-replicas 1 --max-replicas 3 \
    --query "properties.configuration.ingress.fqdn"
done
