#!/bin/bash
set -e

SUBSCRIPTION_NAME="Gobierno de datos"
RESOURCE_GROUP="GRPANALITICA"
LOCATION="eastus"
ACR_NAME="mlopstestacr"
ENVIRONMENT_NAME="mlopstestenvironment"

echo "=== Configurando Azure CLI ==="
az account set --subscription "$SUBSCRIPTION_NAME"
echo "Suscripción activa: $(az account show --query name -o tsv)"

echo ""
echo "=== Verificando/Creando Resource Group ==="
if az group exists --name "$RESOURCE_GROUP" | grep -q "true"; then
    echo "Resource Group '$RESOURCE_GROUP' ya existe."
else
    echo "Creando Resource Group '$RESOURCE_GROUP'..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
    echo "Resource Group creado."
fi

echo ""
echo "=== Verificando/Creando Azure Container Registry ==="
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "ACR '$ACR_NAME' ya existe."
else
    echo "Creando ACR '$ACR_NAME'..."
    az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$ACR_NAME" \
        --sku Basic \
        --admin-enabled true
    echo "ACR creado."
fi

echo ""
echo "=== Verificando/Creando Container App Environment ==="
if az containerapp env show --name "$ENVIRONMENT_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "Container App Environment '$ENVIRONMENT_NAME' ya existe."
else
    echo "Creando Container App Environment '$ENVIRONMENT_NAME'..."
    az containerapp env create \
        --name "$ENVIRONMENT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION"
    echo "Container App Environment creado."
fi

echo ""
echo "=== Setup completado exitosamente ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "ACR: $ACR_NAME"
echo "Environment: $ENVIRONMENT_NAME"

