#!/bin/bash

echo "=== Verificación de Docker en WSL ==="
echo ""

WSL_DISTRO=$(wslpath -u "$(wslvar USERPROFILE)" 2>/dev/null || echo "desconocida")
CURRENT_DISTRO=$(cat /etc/os-release 2>/dev/null | grep "^NAME=" | cut -d'"' -f2 || echo "desconocida")

echo "Distribución WSL actual: $CURRENT_DISTRO"
echo ""

echo "1. Verificando comando 'docker'..."
if command -v docker &> /dev/null; then
    echo "   ✓ Comando docker encontrado"
    DOCKER_VERSION=$(docker --version 2>/dev/null)
    echo "   Versión: $DOCKER_VERSION"
else
    echo "   ✗ Comando docker NO encontrado"
    echo ""
    echo "   ACCIÓN REQUERIDA:"
    echo "   - Abre Docker Desktop en Windows"
    echo "   - Settings > Resources > WSL Integration"
    echo "   - Activa la integración para esta distribución"
    echo "   - Reinicia esta terminal"
    exit 1
fi

echo ""
echo "2. Verificando que Docker esté corriendo..."
if docker info &> /dev/null; then
    echo "   ✓ Docker está corriendo y accesible"
    DOCKER_INFO=$(docker info 2>/dev/null | grep "Operating System" | head -1 || echo "")
    if [ ! -z "$DOCKER_INFO" ]; then
        echo "   $DOCKER_INFO"
    fi
else
    echo "   ✗ Docker no está corriendo o no es accesible"
    echo ""
    echo "   ACCIÓN REQUERIDA:"
    echo "   - Asegúrate de que Docker Desktop esté corriendo en Windows"
    echo "   - Verifica que la integración WSL esté activada"
    echo "   - Intenta reiniciar Docker Desktop"
    exit 1
fi

echo ""
echo "3. Verificando acceso a Docker daemon..."
if docker ps &> /dev/null; then
    echo "   ✓ Puedes comunicarte con el daemon de Docker"
else
    echo "   ✗ No puedes comunicarte con el daemon"
    echo ""
    echo "   ACCIÓN REQUERIDA:"
    echo "   - Verifica permisos de usuario"
    echo "   - Reinicia Docker Desktop"
    exit 1
fi

echo ""
echo "=== Verificación completada ==="
echo "✓ Docker está correctamente configurado y funcionando"
echo ""
echo "Puedes ejecutar './deploy.sh' para desplegar tu aplicación."

