#!/usr/bin/env bash
set -euo pipefail

# Despliega la imagen en Azure Container Apps para entornos dev y staging.

RESOURCE_GROUP=${RESOURCE_GROUP:-GRPANALITICA}
CONTAINERAPPS_ENV=${CONTAINERAPPS_ENV:-mlops-cae}
ACR_NAME=${ACR_NAME:?"Defina ACR_NAME"}
IMAGE_NAME=${IMAGE_NAME:?"Defina IMAGE_NAME (loginServer/repo:tag)"}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:?"Defina STORAGE_ACCOUNT"}
MODEL_CONTAINER=${MODEL_CONTAINER:-models}
ENVIRONMENTS=${ENVIRONMENTS:-"dev staging"}
CPU=${CPU:-0.5}
MEMORY=${MEMORY:-1.0Gi}
MIN_REPLICAS=${MIN_REPLICAS:-1}
MAX_REPLICAS=${MAX_REPLICAS:-3}

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
CONNECTION_STRING=$(az storage account show-connection-string --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" --query connectionString -o tsv)

for ENV_NAME in $ENVIRONMENTS; do
  APP_NAME="mlops-${ENV_NAME}-api"
  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENV" \
    --image "$IMAGE_NAME" \
    --ingress external \
    --target-port 8080 \
    --registry-server "$ACR_LOGIN_SERVER" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --env-vars \
      AZURE_STORAGE_CONNECTION_STRING="$CONNECTION_STRING" \
      AZURE_MODEL_CONTAINER="$MODEL_CONTAINER" \
      APP_ENV="$ENV_NAME" \
    --tags project=mlops env="$ENV_NAME" \
    --query properties.configuration.ingress.fqdn -o tsv
  echo "Aplicación desplegada: $APP_NAME"
done
