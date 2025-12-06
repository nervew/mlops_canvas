#!/usr/bin/env bash
set -euo pipefail

# Required environment variables
: "${STORAGE_ACCOUNT:?Debe definir STORAGE_ACCOUNT}"
: "${MODEL_NAME:?Debe definir MODEL_NAME}"

MODEL_CONTAINER=${MODEL_CONTAINER:-models}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"artifacts/${MODEL_NAME}"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"mlops-rg"}

MODEL_FILE=${MODEL_FILE:-"${ARTIFACTS_DIR}/model/${MODEL_NAME}.pickle"}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-"${ARTIFACTS_DIR}/requirements/requirements.txt"}
SCHEMA_FILE=${SCHEMA_FILE:-"${ARTIFACTS_DIR}/data/schema.empty.parquet"}

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "No se encontró el modelo en $MODEL_FILE" >&2
  exit 1
fi

az storage blob upload \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$MODEL_CONTAINER" \
  --file "$MODEL_FILE" \
  --name "${MODEL_NAME}/model/$(basename "$MODEL_FILE")" \
  --auth-mode login

test -f "$REQUIREMENTS_FILE" && az storage blob upload \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$MODEL_CONTAINER" \
  --file "$REQUIREMENTS_FILE" \
  --name "${MODEL_NAME}/requirements/requirements.txt" \
  --auth-mode login

if [[ -f "$SCHEMA_FILE" ]]; then
  az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$MODEL_CONTAINER" \
    --file "$SCHEMA_FILE" \
    --name "${MODEL_NAME}/data/$(basename "$SCHEMA_FILE")" \
    --auth-mode login
fi

echo "Artefactos publicados en container $MODEL_CONTAINER bajo el prefijo $MODEL_NAME/"
