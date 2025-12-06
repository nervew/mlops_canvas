#!/usr/bin/env bash
set -euo pipefail

RG_NAME=${RG_NAME:-rg-analitica-datamlops-dev-mlops}
LOCATION=${LOCATION:-eastus}
ACR_NAME=${ACR_NAME:-mlopsanalitica}
PLAN_NAME=${PLAN_NAME:-ml-plan}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-mlopsfuncstor}
TRAIN_APP=${TRAIN_APP:-train-func}
PRED_APP=${PRED_APP:-predict-func}
TRAIN_IMAGE=${TRAIN_IMAGE:-training-api:latest}
PRED_IMAGE=${PRED_IMAGE:-inference-api:latest}

LOGIN_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)

az group create -n "$RG_NAME" -l "$LOCATION"
az acr create -n "$ACR_NAME" -g "$RG_NAME" --sku Basic || true
az acr login -n "$ACR_NAME"

# La cuenta de Storage es obligatoria para Function Apps.
az storage account create \
  -n "$STORAGE_ACCOUNT" \
  -g "$RG_NAME" \
  -l "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2

az functionapp plan create -g "$RG_NAME" -n "$PLAN_NAME" --location "$LOCATION" --number-of-workers 1 --sku EP1 --is-linux

az functionapp create -g "$RG_NAME" -p "$PLAN_NAME" -n "$TRAIN_APP" -s "$STORAGE_ACCOUNT" --runtime custom --deployment-container-image-name "$LOGIN_SERVER/$TRAIN_IMAGE"
az functionapp create -g "$RG_NAME" -p "$PLAN_NAME" -n "$PRED_APP" -s "$STORAGE_ACCOUNT" --runtime custom --deployment-container-image-name "$LOGIN_SERVER/$PRED_IMAGE"

for APP in "$TRAIN_APP" "$PRED_APP"; do
  # Seleccionar la imagen correcta según la app
  IMAGE="$TRAIN_IMAGE"
  if [[ "$APP" == "$PRED_APP" ]]; then
    IMAGE="$PRED_IMAGE"
  fi

  az functionapp config appsettings set -g "$RG_NAME" -n "$APP" --settings "WEBSITES_PORT=8080" "MODEL_DIR=/models"
  az functionapp identity assign -g "$RG_NAME" -n "$APP"
  az resource update --ids "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RG_NAME/providers/Microsoft.Web/sites/$APP/config/appsettings" --set properties.DOCKER_REGISTRY_SERVER_URL="https://$LOGIN_SERVER"
  az webapp config container set -g "$RG_NAME" -n "$APP" --docker-custom-image-name "$LOGIN_SERVER/$IMAGE" --docker-registry-server-url "https://$LOGIN_SERVER"
done

