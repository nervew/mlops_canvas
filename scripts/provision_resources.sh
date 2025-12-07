#!/usr/bin/env bash
set -euo pipefail

SUBSCRIPTION=${SUBSCRIPTION:-"Gobierno de datos"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"GRPANALITICA"}
LOCATION=${LOCATION:-"eastus"}
PREFIX=${PREFIX:-"mlopstest"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"mlopstestacr.azurecr.io/mlopstest-api:v1"}

echo "Setting subscription to ${SUBSCRIPTION}"
az account set --subscription "${SUBSCRIPTION}"

echo "Creating resource group ${RESOURCE_GROUP} (${LOCATION}) if missing..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output none

echo "Deploying Bicep template..."
az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file "$(dirname "$0")/../infra/provision.bicep" \
  --parameters location="${LOCATION}" prefix="${PREFIX}" containerImage="${CONTAINER_IMAGE}" \
  --output table
