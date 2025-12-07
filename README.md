# mlops_canvas

Aplicación FastAPI mínima empaquetada en contenedor y desplegable en Azure Container Apps usando scripts de línea de comando (sin Terraform).

## Requisitos
- Azure CLI con extensión `containerapp` y sesión activa (`az login`).
- Permisos para crear Resource Groups, ACR, Log Analytics y Container Apps.
- Docker para construir y publicar la imagen.
- Bash 4+.

## Variables principales
| Variable | Descripción | Valor por defecto |
| --- | --- | --- |
| `PREFIX` | Prefijo base para nombrar recursos. | `mlops` |
| `LOCATION` | Región de Azure. | `eastus` |
| `RESOURCE_GROUP` | Grupo de recursos. | `${PREFIX}-rg` |
| `ACR_NAME` | Azure Container Registry. | `${PREFIX}acr` |
| `LOG_WORKSPACE` | Log Analytics Workspace. | `${PREFIX}-logs` |
| `CONTAINERAPPS_ENV` | Environment de Container Apps. | `${PREFIX}-env` |
| `CONTAINERAPP_NAME` | Nombre de la app. | `${PREFIX}-api` |
| `IMAGE_NAME` / `IMAGE_TAG` | Nombre y tag de la imagen. | `${PREFIX}-api` / `latest` |
| `CONTAINER_PORT` | Puerto expuesto por la app. | `8000` |

## Preparar la infraestructura
Ejecuta una sola vez para crear o validar recursos necesarios:
```bash
az login
./setup.sh
```

## Desplegar la aplicación
Construye la imagen local, publícala en ACR y crea/actualiza la Container App:
```bash
./deploy.sh
```

## Notas
- Ambos scripts son idempotentes y reutilizan recursos existentes cuando es posible.
- Ajusta las variables de entorno si necesitas nombres o regiones distintas.
- La aplicación escucha en el puerto definido por `CONTAINER_PORT` (8000 por defecto).
