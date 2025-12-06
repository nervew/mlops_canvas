#!/usr/bin/env bash
set -euo pipefail

# ======== CONFIG ========
RG_NAME=${RG_NAME:-rg-analitica-datamlops-dev-mlops}
LOCATION=${LOCATION:-eastus}
ACR_NAME=${ACR_NAME:-mlopsanalitica}
ENV_NAME=${ENV_NAME:-mlops-env}

TRAIN_APP_NAME=${TRAIN_APP_NAME:-training-api-app}
PRED_APP_NAME=${PRED_APP_NAME:-inference-api-app}

TRAIN_IMAGE_TAG=${TRAIN_IMAGE_TAG:-training-api:latest}
PRED_IMAGE_TAG=${PRED_IMAGE_TAG:-inference-api:latest}

# ======== CREATE ENVIRONMENT ========
echo "📦 Creating Container Apps environment..."
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RG_NAME" \
  --location "$LOCATION"

# ======== GET ACR INFO ========
echo "🔍 Getting ACR credentials..."
LOGIN_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query passwords[0].value -o tsv)

# ======== CREATE TRAINING API ========
echo "🚀 Deploying Training API..."
az containerapp create \
  --name "$TRAIN_APP_NAME" \
  --resource-group "$RG_NAME" \
  --environment "$ENV_NAME" \
  --ingress external \
  --target-port 8000 \
  --image "$LOGIN_SERVER/$TRAIN_IMAGE_TAG" \
  --registry-server "$LOGIN_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS"

# ======== CREATE INFERENCE API ========
echo "🚀 Deploying Inference API..."
az containerapp create \
  --name "$PRED_APP_NAME" \
  --resource-group "$RG_NAME" \
  --environment "$ENV_NAME" \
  --ingress external \
  --target-port 8000 \
  --image "$LOGIN_SERVER/$PRED_IMAGE_TAG" \
  --registry-server "$LOGIN_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS"

# ======== OUTPUT URLS ========
TRAIN_URL=$(az containerapp show -n "$TRAIN_APP_NAME" -g "$RG_NAME" --query properties.configuration.ingress.fqdn -o tsv)
PRED_URL=$(az containerapp show -n "$PRED_APP_NAME" -g "$RG_NAME" --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "======================================================"
echo " 🎉 APIs deployed successfully!"
echo "======================================================"
echo "Training API → https://$TRAIN_URL"
echo "Inference API → https://$PRED_URL"
echo ""