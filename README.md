# MLOps Canvas APIs

Este proyecto contiene dos APIs basadas en FastAPI empaquetadas como Azure Functions en contenedores personalizados: una para entrenamiento AutoML con FLAML y Optuna, y otra para inferencia de modelos serializados.

## Estructura
- `training_api/`: API de entrenamiento.
- `inference_api/`: API de inferencia.
- `infra/azure`: scripts para crear ACR y desplegar Function Apps.
- `tests/`: pruebas mínimas.

## Uso local
```bash
make test
```

Construir imágenes:
```bash
make build
```

Publicar en Azure:
```bash
./infra/azure/create_acr_and_push.sh
./infra/azure/deploy_functions.sh
```
