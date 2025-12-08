#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="mlops-canvas-api"
CONTAINER_NAME="mlops-canvas-api-local"
PORT=8000

echo "=== Construyendo imagen Docker ==="
docker build -t $IMAGE_NAME .

echo ""
echo "=== Deteniendo contenedor existente (si existe) ==="
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo ""
echo "=== Ejecutando contenedor ==="
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  $IMAGE_NAME

echo ""
echo "=== Contenedor iniciado ==="
echo "API disponible en: http://localhost:$PORT"
echo "Documentación en: http://localhost:$PORT/docs"
echo ""
echo "Para ver los logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Para detener el contenedor:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "Esperando a que el contenedor esté listo..."
sleep 3

echo "Verificando estado del contenedor..."
sleep 2
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✓ Contenedor está corriendo"
    echo ""
    echo "Puedes probar la API con:"
    echo "  curl http://localhost:$PORT/health"
    echo "  curl http://localhost:$PORT/"
else
    echo "⚠ Error: El contenedor no está corriendo"
    echo "Revisa los logs con: docker logs $CONTAINER_NAME"
    exit 1
fi

