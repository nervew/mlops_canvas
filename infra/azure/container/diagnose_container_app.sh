#!/usr/bin/env bash
set -euo pipefail

APP_NAME=$1
RG_NAME=$2

echo ""
echo "=============================="
echo " 🔍 Diagnóstico Container App"
echo "=============================="
echo "App: $APP_NAME"
echo "RG:  $RG_NAME"
echo ""

# 1. Estado general de la App
echo "📌 1) Estado General de la Container App"
az containerapp show -n "$APP_NAME" -g "$RG_NAME" -o table
echo ""

# 2. Revisiones activas/inactivas
echo "📌 2) Revisiones (si hay CrashLoop aparece aquí)"
az containerapp revision list -n "$APP_NAME" -g "$RG_NAME" -o table
echo ""

# 3. Logs del contenedor
echo "📌 3) Logs del contenedor (últimos eventos)"
az containerapp logs show \
  -n "$APP_NAME" \
  -g "$RG_NAME" \
  --type container \
  --tail 50
echo ""

# 4. Logs del sistema (errores de infraestructura)
echo "📌 4) Logs del sistema"
az containerapp logs show \
  -n "$APP_NAME" \
  -g "$RG_NAME" \
  --type system \
  --tail 50
echo ""

# 5. Verificar si la app expone correctamente el puerto
echo "📌 5) Puerto configurado en ingress"
PORT=$(az containerapp show -n "$APP_NAME" -g "$RG_NAME" --query properties.configuration.ingress.targetPort -o tsv)
echo "    Puerto configurado: $PORT"
echo ""

# 6. Verificar si el contenedor usa otra imagen
echo "📌 6) Imagen desplegada"
az containerapp show -n "$APP_NAME" -g "$RG_NAME" --query properties.template.containers
echo ""

# 7. Validar configuración del registry
echo "📌 7) Configuración del Registry"
a
