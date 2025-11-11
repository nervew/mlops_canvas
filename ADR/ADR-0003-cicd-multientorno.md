# ADR-0003 Estrategia CI/CD multi-entorno

## Estado
Aceptado (2024-04-30)

## Contexto
Se busca promover de `dev`→`qa`→`prod` con gates, escaneo de imágenes y aprobación explícita.

## Decisión
GitHub Actions con `ci.yml`, `cd-dev.yml`, `cd-qa.yml`, `cd-prod.yml` y acciones reutilizables (Trivy, despliegue Terraform/Helm) vía identidades federadas.

## Alternativas
- **Azure DevOps Pipelines**: integración profunda pero doble mantenimiento.
- **Jenkins**: flexible, requiere hosting propio.

## Consecuencias
+ Acciones reutilizables, configuración declarativa.
− Límites de minutos y dependencia de OIDC.

## Seguimiento
Revisar duración de pipelines trimestralmente y asegurar smoke tests post-despliegue antes de cerrar PR.
