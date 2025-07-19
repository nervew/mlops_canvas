# Evaluación y Optimización de Pipelines MLOps en Azure

Este documento resume desafíos comunes en pipelines de Machine Learning y propone
estrategias para optimizar su ejecución en Azure Databricks.

## Evaluación actual
- Ausencia de control de versiones de datos y modelos.
- Bajos niveles de automatización en CI/CD.
- Escasa observabilidad de experimentos y fallos.

## Estrategia de optimización
1. **Herramientas**: DVC para versionar datos, MLflow para experimentos,
   Azure ML o Databricks Workflows para orquestación y despliegue.
2. **Gestión de dependencias**: uso de Poetry para entornos reproducibles.
3. **Calidad de código**: integración de Ruff, Black y mypy.
4. **Seguridad**: almacenar credenciales en Azure Key Vault y usar Azure
   Monitor para auditoría.

## Implementación práctica
El repositorio se organiza en:

```
src/          # Código fuente del pipeline
configs/      # Archivos YAML de configuración
models/       # Artefactos de modelo
notebooks/    # Exploraciones y experimentos
tests/        # Pruebas automáticas
```

Los conectores implementan el patrón Ports & Adapters para desacoplar la
lógica de negocio de las bases de datos.

## Recomendaciones finales
- Escalar la solución con autoescalado de clusters.
- Reducir latencia cacheando modelos y usando endpoints serverless.
- Mejorar resiliencia con monitoreo continuo y alertas automáticas.
