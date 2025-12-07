# API Hola Mundo en Azure Container Apps

Infraestructura automatizada para desplegar una API "hola mundo" en Azure usando Container Registry y Container Apps.

## Recursos Azure

- **Resource Group**: `GRPANALITICA`
- **Suscripción**: `Gobierno de datos`
- **Container Registry**: `mlopstestacr`
- **Container App Environment**: `mlopstestenvironment`
- **Container App**: `mlopstestapp`

## Dependencias

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- [Docker](https://docs.docker.com/get-docker/)
- Permisos de Contributor en la suscripción "Gobierno de datos"

## Estructura del proyecto

```
.
├── app.py              # API FastAPI
├── requirements.txt    # Dependencias Python
├── Dockerfile          # Imagen Docker
├── setup.sh           # Provisiona recursos en Azure
├── deploy.sh          # Despliega la aplicación
└── README.md          # Este archivo
```

## Uso

### 1. Provisionar infraestructura

Crea los recursos necesarios en Azure (idempotente):

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Desplegar aplicación

Construye, pushea y despliega la app:

```bash
chmod +x deploy.sh
./deploy.sh
```

Al finalizar, obtendrás la URL pública de tu API.

### 3. Probar la API

```bash
curl https://<tu-app-url>/
curl https://<tu-app-url>/health
```

## Endpoints

- `GET /` - Retorna `{"message": "hola mundo"}`
- `GET /health` - Health check

## Notas

- Los scripts son idempotentes: puedes ejecutarlos múltiples veces
- `deploy.sh` actualiza la app si ya existe
- La app usa recursos mínimos (0.25 CPU, 0.5 GB RAM)

