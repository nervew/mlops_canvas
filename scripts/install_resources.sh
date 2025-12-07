#!/usr/bin/env bash
set -euo pipefail

# Instala y configura los recursos base para el pipeline de inferencia en Azure.
# Crea (si no existen) el resource group GRPANALITICA y los componentes con prefijo mlops.

RESOURCE_GROUP=${RESOURCE_GROUP:-GRPANALITICA}
LOCATION=${LOCATION:-eastus}
PREFIX=${PREFIX:-mlops}
SUFFIX=${SUFFIX:-$(openssl rand -hex 2)}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-${PREFIX}stor${SUFFIX}}
ACR_NAME=${ACR_NAME:-${PREFIX}acr${SUFFIX}}
CONTAINERAPPS_ENV=${CONTAINERAPPS_ENV:-${PREFIX}-cae}
LOG_ANALYTICS_WORKSPACE=${LOG_ANALYTICS_WORKSPACE:-${PREFIX}-laws}
MODEL_CONTAINER=${MODEL_CONTAINER:-models}

az config set extension.use_dynamic_install=yes_without_prompt >/dev/null

az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

# Log Analytics para Container Apps
LA_WORKSPACE_ID=$(az monitor log-analytics workspace create \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
  --location "$LOCATION" \
  --query id -o tsv)
LA_WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
  --query primarySharedKey -o tsv)

# Azure Container Apps Environment
az containerapp env create \
  --name "$CONTAINERAPPS_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --logs-workspace-id "$LA_WORKSPACE_ID" \
  --logs-workspace-key "$LA_WORKSPACE_KEY" \
  --location "$LOCATION" \
  --tags project=mlops env=shared

# Azure Container Registry
az acr create --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --sku Standard --location "$LOCATION"
az acr update --name "$ACR_NAME" --admin-enabled true

# Storage account y contenedor de modelos
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false

az storage container create \
  --name "$MODEL_CONTAINER" \
  --account-name "$STORAGE_ACCOUNT" \
  --auth-mode login \
  --public-access off

echo "Recursos listos:" \
  "\n- Resource Group: $RESOURCE_GROUP" \
  "\n- Storage Account: $STORAGE_ACCOUNT (contenedor: $MODEL_CONTAINER)" \
  "\n- ACR: $ACR_NAME" \
  "\n- Container Apps Environment: $CONTAINERAPPS_ENV"
