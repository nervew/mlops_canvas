#!/usr/bin/env bash
set -euo pipefail

SUBSCRIPTION=${SUBSCRIPTION:-"Gobierno de datos"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"GRPANALITICA"}
LOCATION=${LOCATION:-"eastus2"}
PREFIX=${PREFIX:-"mlopstest"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"mlopstestacr.azurecr.io/mlopstest-api:v1"}

echo "Setting subscription to ${SUBSCRIPTION}"
az account set --subscription "${SUBSCRIPTION}"

echo "Checking if resource group ${RESOURCE_GROUP} exists..."
if az group show --name "${RESOURCE_GROUP}" >/dev/null 2>&1; then
    echo "Resource group ${RESOURCE_GROUP} already exists. Skipping creation."
else
    echo "Creating resource group ${RESOURCE_GROUP} in ${LOCATION}..."
    az group create \
        --name "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --output none
fi

echo "Deploying Bicep template..."
az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file "$(dirname "$0")/../infra/provision.bicep" \
  --parameters location="${LOCATION}" prefix="${PREFIX}" containerImage="${CONTAINER_IMAGE}" \
  --output table
