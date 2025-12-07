#!/usr/bin/env bash
set -euo pipefail

SUBSCRIPTION=${SUBSCRIPTION:-"Gobierno de datos"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"GRPANALITICA"}
PREFIX=${PREFIX:-"mlopstest"}
IMAGE=${IMAGE:-"mlopstestacr.azurecr.io/mlopstest-api:v1"}

APP_NAME="${PREFIX}-api"

az account set --subscription "${SUBSCRIPTION}"

echo "Updating Container App ${APP_NAME} with image ${IMAGE}..."
az containerapp update \
  --name "${APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --image "${IMAGE}" \
  --output table

echo "Done. Review ingress URL with: az containerapp show --name ${APP_NAME} --resource-group ${RESOURCE_GROUP} --query properties.configuration.ingress.fqdn"
