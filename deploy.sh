#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=== Verificando dependencias ==="
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker no está disponible en esta distribución WSL."
    echo ""
    echo "Para solucionarlo:"
    echo "1. Abre Docker Desktop en Windows"
    echo "2. Ve a Settings > Resources > WSL Integration"
    echo "3. Activa la integración para esta distribución WSL"
    echo "4. Reinicia esta terminal o ejecuta: source ~/.bashrc"
    echo ""
    echo "Más información: https://docs.docker.com/go/wsl2/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker no está corriendo o no tienes permisos."
    echo ""
    echo "Verifica que:"
    echo "1. Docker Desktop esté corriendo en Windows"
    echo "2. La integración WSL esté activada para esta distribución"
    exit 1
fi

echo "Docker está disponible ✓"

echo ""
echo "=== Configurando Azure CLI ==="
az account set --subscription "$SUBSCRIPTION_NAME"

echo ""
echo "=== Construyendo imagen Docker ==="
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "=== Login a Azure Container Registry ==="
az acr login --name "$ACR_NAME"

echo ""
echo "=== Obteniendo login server del ACR ==="
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)
echo "ACR Login Server: $ACR_LOGIN_SERVER"

echo ""
echo "=== Taggeando imagen para ACR ==="
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "=== Pusheando imagen a ACR ==="
docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "=== Obteniendo credenciales del ACR ==="
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo ""
echo "=== Desplegando/Actualizando Container App ==="
if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "Container App '$APP_NAME' existe. Actualizando..."
    az containerapp update \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Container App actualizada."
else
    echo "Creando Container App '$APP_NAME'..."
    az containerapp create \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --environment "$ENVIRONMENT_NAME" \
        --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
        --target-port "$TARGET_PORT" \
        --ingress "$INGRESS" \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USERNAME" \
        --registry-password "$ACR_PASSWORD" \
        --cpu "$CPU" \
        --memory "$MEMORY"
    echo "Container App creada."
fi

echo ""
echo "=== Obteniendo URL de la aplicación ==="
APP_URL=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "=== Despliegue completado exitosamente ==="
echo "Aplicación disponible en: https://${APP_URL}"
echo ""
echo "Prueba tu API con:"
echo "  curl https://${APP_URL}/"
echo "  curl https://${APP_URL}/health"
echo "  curl https://${APP_URL}/model-info"
echo "  curl -X POST https://${APP_URL}/predict -H 'Content-Type: application/json' -d '{\"feature_1\": 1.0}'"

