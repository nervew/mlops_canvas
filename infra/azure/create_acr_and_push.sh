#!/usr/bin/env bash
set -euo pipefail

RG_NAME=${RG_NAME:-rg-ml-apis}
LOCATION=${LOCATION:-eastus}
ACR_NAME=${ACR_NAME:-mlacr$RANDOM}
TRAIN_IMAGE_TAG=${TRAIN_IMAGE_TAG:-training-api:latest}
PRED_IMAGE_TAG=${PRED_IMAGE_TAG:-inference-api:latest}

az group create -n "$RG_NAME" -l "$LOCATION"
az acr create -n "$ACR_NAME" -g "$RG_NAME" --sku Basic
az acr login -n "$ACR_NAME"
LOGIN_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)

docker buildx build -t "$LOGIN_SERVER/$TRAIN_IMAGE_TAG" ./training_api --push

docker buildx build -t "$LOGIN_SERVER/$PRED_IMAGE_TAG" ./inference_api --push

echo "Images pushed to $LOGIN_SERVER"
