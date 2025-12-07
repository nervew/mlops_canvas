#!/usr/bin/env bash
set -euo pipefail

PREFIX=${PREFIX:-"mlops"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"${PREFIX}-rg"}
ACR_NAME=${ACR_NAME:-"${PREFIX}acr"}
CONTAINERAPPS_ENV=${CONTAINERAPPS_ENV:-"${PREFIX}-env"}
CONTAINERAPP_NAME=${CONTAINERAPP_NAME:-"${PREFIX}-api"}
IMAGE_NAME=${IMAGE_NAME:-"${PREFIX}-api"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
CONTAINER_PORT=${CONTAINER_PORT:-8000}
CPU=${CPU:-1.0}
MEMORY=${MEMORY:-"2Gi"}

command -v az >/dev/null 2>&1 || { echo "Azure CLI (az) es requerido" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker es requerido para construir la imagen" >&2; exit 1; }

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query "passwords[0].value" -o tsv)
FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name "$ACR_NAME"

echo "Construyendo imagen $FULL_IMAGE..."
docker build -t "$FULL_IMAGE" .

echo "Enviando imagen a ACR..."
docker push "$FULL_IMAGE"

if az containerapp show --name "$CONTAINERAPP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "Actualizando Container App existente..."
  az containerapp update \
    --name "$CONTAINERAPP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$FULL_IMAGE" \
    --set-env-vars APP_ENV=production \
    --output table
else
  echo "Creando Container App..."
  az containerapp create \
    --name "$CONTAINERAPP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENV" \
    --image "$FULL_IMAGE" \
    --target-port "$CONTAINER_PORT" \
    --ingress external \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --set-env-vars APP_ENV=production \
    --output table
fi

echo "Despliegue completado. Imagen activa: $FULL_IMAGE"
