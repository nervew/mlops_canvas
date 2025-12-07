#!/bin/bash

set -e

echo "=== Paso 1: Generar dataset ==="
python generate_timeseries_data.py

echo ""
echo "=== Paso 2: Entrenar modelo ==="
python train_model.py

echo ""
echo "=== Paso 3: Subir a Azure ==="
bash upload_to_azure.sh

echo ""
echo "=== Pipeline completado ==="

