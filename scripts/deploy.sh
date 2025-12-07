#!/usr/bin/env bash
set -euo pipefail

PREFIX=${PREFIX:-"mlopstest"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"GRPANALITICA"}
LOCATION=${LOCATION:-"eastus"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

pushd "$PROJECT_DIR" >/dev/null

terraform init -input=false
terraform apply -input=false -auto-approve \
  -var "prefix=${PREFIX}" \
  -var "resource_group_name=${RESOURCE_GROUP}" \
  -var "location=${LOCATION}"

ACR_LOGIN_SERVER=$(terraform output -raw acr_login_server)
CONTAINER_APP_NAME=$(terraform output -raw container_app_name)

ACR_NAME=${ACR_LOGIN_SERVER%%.*}
IMAGE_NAME="${PREFIX}-api"
FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name "$ACR_NAME"

docker build -t "$FULL_IMAGE" .
docker push "$FULL_IMAGE"

az containerapp update \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$FULL_IMAGE"

popd >/dev/null
