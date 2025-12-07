#!/bin/bash

set -e

RESOURCE_GROUP="GRPANALITICA"
STORAGE_ACCOUNT_NAME="mlopstestanalitica"
CONTAINER_NAME="mlopstestupload"
MODEL_NAME="nombre_model"
LOCATION="westeurope"

echo "=== Configuración ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo "Container: $CONTAINER_NAME"
echo "Model Name: $MODEL_NAME"
echo ""

STORAGE_EXISTS=$(az storage account show --resource-group $RESOURCE_GROUP --name $STORAGE_ACCOUNT_NAME --query "name" -o tsv 2>/dev/null || echo "")

if [ -z "$STORAGE_EXISTS" ]; then
    echo "Storage account '$STORAGE_ACCOUNT_NAME' no existe en el resource group '$RESOURCE_GROUP'. Creándolo..."
    
    set +e
    CREATE_OUTPUT=$(az storage account create \
        --resource-group $RESOURCE_GROUP \
        --name $STORAGE_ACCOUNT_NAME \
        --location $LOCATION \
        --sku Standard_LRS 2>&1)
    CREATE_EXIT_CODE=$?
    set -e
    
    if [ $CREATE_EXIT_CODE -ne 0 ]; then
        if echo "$CREATE_OUTPUT" | grep -q "StorageAccountAlreadyTaken"; then
            echo ""
            echo "Error: El storage account '$STORAGE_ACCOUNT_NAME' ya existe en Azure (pero no en el resource group '$RESOURCE_GROUP')."
            echo "Los nombres de storage accounts deben ser únicos globalmente."
            echo ""
            echo "Opciones:"
            echo "  1. Usa un nombre diferente para el storage account"
            echo "  2. Si el storage account existe en otro resource group, muévelo o úsalo desde ahí"
            exit 1
        else
            echo "Error al crear el storage account:"
            echo "$CREATE_OUTPUT"
            exit 1
        fi
    fi
    echo "Storage account '$STORAGE_ACCOUNT_NAME' creado exitosamente"
else
    echo "Storage account '$STORAGE_ACCOUNT_NAME' existe en el resource group '$RESOURCE_GROUP'. Utilizándolo..."
fi

echo ""
echo "=== Obtener clave de acceso ==="
STORAGE_KEY=$(az storage account keys list \
    --resource-group $RESOURCE_GROUP \
    --account-name $STORAGE_ACCOUNT_NAME \
    --query "[0].value" -o tsv)

echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo ""

echo "=== Verificar/Crear contenedor ==="
CONTAINER_EXISTS=$(az storage container exists \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY \
    --name $CONTAINER_NAME \
    --query "exists" -o tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Creando contenedor: $CONTAINER_NAME"
    az storage container create \
        --account-name $STORAGE_ACCOUNT_NAME \
        --account-key $STORAGE_KEY \
        --name $CONTAINER_NAME \
        --public-access off
else
    echo "Contenedor $CONTAINER_NAME ya existe"
fi

echo ""
echo "=== Subiendo archivos ==="

if [ ! -f "model/modelo_entrenado.pkl" ]; then
    echo "Error: No se encuentra model/modelo_entrenado.pkl"
    exit 1
fi

if [ ! -f "requirement/requirements.txt" ]; then
    echo "Error: No se encuentra requirement/requirements.txt"
    exit 1
fi

if [ ! -f "csv/dataset.csv" ]; then
    echo "Error: No se encuentra csv/dataset.csv"
    exit 1
fi

echo "Subiendo modelo..."
az storage blob upload \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY \
    --container-name $CONTAINER_NAME \
    --name "$MODEL_NAME/model/modelo_entrenado.pkl" \
    --file "model/modelo_entrenado.pkl" \
    --overwrite

echo "Subiendo requirements.txt..."
az storage blob upload \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY \
    --container-name $CONTAINER_NAME \
    --name "$MODEL_NAME/requirement/requirements.txt" \
    --file "requirement/requirements.txt" \
    --overwrite

echo "Subiendo dataset.csv..."
az storage blob upload \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY \
    --container-name $CONTAINER_NAME \
    --name "$MODEL_NAME/csv/dataset.csv" \
    --file "csv/dataset.csv" \
    --overwrite

echo ""
echo "=== Verificación ==="
echo "Listando archivos subidos:"
az storage blob list \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY \
    --container-name $CONTAINER_NAME \
    --prefix "$MODEL_NAME/" \
    --query "[].name" -o table

echo ""
echo "=== Completado ==="
echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo "Container: $CONTAINER_NAME"
echo "Estructura: $MODEL_NAME/{model,requirement,csv}/"

