#!/bin/bash
set -e

SUBSCRIPTION_NAME="Gobierno de datos"
RESOURCE_GROUP="GRPANALITICA"
ACR_NAME="mlopstestacr"
ENVIRONMENT_NAME="mlopstestenvironment"
APP_NAME="mlopstestapp"
IMAGE_NAME="hola-mundo-api"
IMAGE_TAG="latest"

USE_ACR_BUILD="${USE_ACR_BUILD:-false}"

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
echo "=== Obteniendo login server del ACR ==="
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)
echo "ACR Login Server: $ACR_LOGIN_SERVER"

if [ "$USE_ACR_BUILD" = "true" ]; then
    echo ""
    echo "=== Construyendo imagen directamente en ACR (evita problemas de conectividad) ==="
    az acr build \
        --registry "$ACR_NAME" \
        --image "${IMAGE_NAME}:${IMAGE_TAG}" \
        --file Dockerfile .
    echo "✓ Imagen construida y pusheada exitosamente en ACR"
else
    echo ""
    echo "=== Construyendo nueva imagen Docker localmente ==="
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

    echo ""
    echo "=== Login a Azure Container Registry ==="
    az acr login --name "$ACR_NAME"

    echo ""
    echo "=== Taggeando imagen para ACR ==="
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

    echo ""
    echo "=== Pusheando imagen a ACR (reemplazando imagen anterior) ==="
    MAX_RETRIES=3
    RETRY_COUNT=0
    PUSH_SUCCESS=false

    set +e
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" 2>&1; then
            PUSH_SUCCESS=true
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo ""
                echo "⚠️  Push falló. Reintentando ($RETRY_COUNT/$MAX_RETRIES) en 5 segundos..."
                sleep 5
                echo "Reautenticando en ACR..."
                az acr login --name "$ACR_NAME"
            fi
        fi
    done
    set -e

    if [ "$PUSH_SUCCESS" = false ]; then
        echo ""
        echo "❌ ERROR: No se pudo pushear la imagen después de $MAX_RETRIES intentos."
        echo ""
        echo "💡 Solución: Usa construcción en ACR para evitar problemas de conectividad:"
        echo "   USE_ACR_BUILD=true bash deploy2.sh"
        echo ""
        echo "O verifica:"
        echo "1. Tu conexión a internet"
        echo "2. Accesibilidad del ACR: az acr check-health --name $ACR_NAME"
        exit 1
    fi

    echo "✓ Imagen pusheada exitosamente"
fi

echo ""
echo "=== Obteniendo credenciales del ACR ==="
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo ""
echo "=== Desplegando/Actualizando Container App con nueva imagen ==="
if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "Container App '$APP_NAME' existe. Actualizando con nueva imagen..."
    az containerapp update \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Container App actualizada con nueva imagen."
else
    echo "Creando Container App '$APP_NAME'..."
    az containerapp create \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --environment "$ENVIRONMENT_NAME" \
        --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
        --target-port 8000 \
        --ingress external \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USERNAME" \
        --registry-password "$ACR_PASSWORD" \
        --cpu 0.25 \
        --memory 0.5Gi
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
echo "  curl -X POST https://${APP_URL}/edad -H 'Content-Type: application/json' -d '{\"fecha_nacimiento\": \"1990-01-15\"}'"

