# Azure MLOps Canvas
> Referencia para entrenar, versionar, desplegar y operar modelos de ML en Azure.

## Índice
[Visión](#visión) · [Guía 5 minutos](#guía-5-minutos) · [Rutas](#rutas) · [Mapa](#mapa) · [Próximos pasos](#próximos-pasos)

## Visión
Infraestructura como código, pipelines AML, inferencia online/batch, monitorización y CI/CD con promoción segura entre entornos.

## Guía 5 minutos
```bash
git clone <repo>
cd mlops_canvas
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
pip install mkdocs mkdocs-material && mkdocs serve
```
> **Experta**: `make plan`

## Rutas
- :seedling: [INSTALL](INSTALL.md) · [GETTING_STARTED](GETTING_STARTED.md) · [GLOSSARY](GLOSSARY.md)
- :rocket: [ARCHITECTURE](ARCHITECTURE.md) · ADRs · [OPERATIONS](OPERATIONS.md) · [API_REFERENCE](API/API_REFERENCE.md)

## Mapa
| Directorio | Rol |
|------------|-----|
| `infra/terraform` | IaC red/AML/AKS/ACR/observabilidad |
| `mlops/aml/pipelines` | YAML entrenamiento/batch |
| `src/training` | Preparación, entrenamiento, evaluación |
| `src/inference/online` | FastAPI `/predict`, `/health`, `/metrics` |
| `src/inference/batch` | Job Pandas + validación esquema |
| `src/monitoring` | Drift, calidad, telemetría |
| `tests` | Pytest unit/integración/e2e |
| `docs` | Contenido extendido (mkdocs) |

## Próximos pasos
Consulta seguridad y operaciones antes de desplegar, usa plantillas de issues/PR y registra nuevas decisiones mediante ADRs.
