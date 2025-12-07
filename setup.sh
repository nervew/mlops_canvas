#!/usr/bin/env bash
set -euo pipefail

PREFIX=${PREFIX:-"mlops"}
LOCATION=${LOCATION:-"eastus"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"${PREFIX}-rg"}
ACR_NAME=${ACR_NAME:-"${PREFIX}acr"}
LOG_WORKSPACE=${LOG_WORKSPACE:-"${PREFIX}-logs"}
CONTAINERAPPS_ENV=${CONTAINERAPPS_ENV:-"${PREFIX}-env"}

command -v az >/dev/null 2>&1 || { echo "Azure CLI (az) es requerido" >&2; exit 1; }

az config set extension.use_dynamic_install=yes_without_prompt >/dev/null
az extension show --name containerapp >/dev/null 2>&1 || az extension add --name containerapp
az provider register --namespace Microsoft.App --wait >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait >/dev/null

az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

echo "Asegurando Azure Container Registry ($ACR_NAME)..."
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --sku Basic \
    --admin-enabled true \
    --output none
else
  echo "ACR ya existe, omitiendo creación"
fi

if ! az monitor log-analytics workspace show --workspace-name "$LOG_WORKSPACE" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "Creando Log Analytics Workspace ($LOG_WORKSPACE)..."
  az monitor log-analytics workspace create \
    --workspace-name "$LOG_WORKSPACE" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
else
  echo "Log Analytics Workspace ya existe, omitiendo creación"
fi

WORKSPACE_ID=$(az monitor log-analytics workspace show --workspace-name "$LOG_WORKSPACE" --resource-group "$RESOURCE_GROUP" --query customerId -o tsv)
WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys --workspace-name "$LOG_WORKSPACE" --resource-group "$RESOURCE_GROUP" --query primarySharedKey -o tsv)

echo "Asegurando Container Apps Environment ($CONTAINERAPPS_ENV)..."
if ! az containerapp env show --name "$CONTAINERAPPS_ENV" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$CONTAINERAPPS_ENV" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --logs-workspace-id "$WORKSPACE_ID" \
    --logs-workspace-key "$WORKSPACE_KEY" \
    --output none
else
  echo "Container Apps Environment ya existe, omitiendo creación"
fi

echo "Recursos listos en el grupo $RESOURCE_GROUP"
