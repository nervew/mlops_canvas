# Servicio de Inferencia ML en Azure Container Apps

Servicio de inferencia de modelos de machine learning desplegado en Azure usando Azure Container Registry y Azure Container Apps.

## Arquitectura

El servicio carga dinámicamente modelos ML desde `artifacts/model.pkl`, detecta el framework utilizado (scikit-learn, xgboost, lightgbm, etc.) desde `artifacts/requirements.txt`, e infiere automáticamente las columnas de entrada requeridas y el tipo de salida desde `artifacts/data.parquet`.

## Recursos Azure

- **Resource Group**: `GRPANALITICA`
- **Suscripción**: `Gobierno de datos`
- **Container Registry**: `mlopstestacr`
- **Container App Environment**: `mlopstestenvironment`
- **Container App**: `mlopstestapp`

## Dependencias

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) instalado y configurado
- [Docker](https://docs.docker.com/get-docker/) instalado y funcionando
- Permisos de Contributor en la suscripción "Gobierno de datos"

### Verificación de Docker en WSL 2

Si usas WSL 2:

1. Instala Docker Desktop en Windows
2. Abre Docker Desktop > Settings > Resources > WSL Integration
3. Activa la integración para tu distribución WSL
4. Reinicia la terminal

**Verificación rápida**: Ejecuta `docker info` para verificar que Docker esté disponible.

## Estructura del proyecto

```
.
├── app.py                  # API FastAPI con inferencia dinámica
├── Dockerfile              # Imagen Docker
├── setup.sh               # Provisiona recursos en Azure
├── deploy.sh              # Despliega la aplicación
└── artifacts/
    ├── model.pkl          # Modelo ML serializado
    ├── data.parquet       # Datos de referencia para inferencia
    └── requirements.txt   # Dependencias Python
```

## Uso

### 1. Provisionar infraestructura

Crea los recursos necesarios en Azure (idempotente):

```bash
chmod +x setup.sh
./setup.sh
```

Este script:
- Selecciona la suscripción "Gobierno de datos"
- Crea el resource group `GRPANALITICA` si no existe
- Crea el Azure Container Registry `mlopstestacr` si no existe
- Crea el Container App Environment `mlopstestenvironment` si no existe

### 2. Desplegar aplicación

Construye, pushea y despliega la app:

```bash
chmod +x deploy.sh
./deploy.sh
```

Este script:
- Construye la imagen Docker
- Hace login en el ACR
- Pushea la imagen al ACR
- Crea o actualiza la Container App `mlopstestapp`

Al finalizar, obtendrás la URL pública de tu API.

### 3. Probar la API

```bash
# Health check
curl https://<tu-app-url>/health

# Información del modelo
curl https://<tu-app-url>/model-info

# Predicción
curl -X POST https://<tu-app-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 1.0, "feature_2": 2.0, ...}'
```

## Endpoints

- `GET /` - Información básica de la API y estado del modelo
- `GET /health` - Health check
- `GET /model-info` - Información sobre el modelo cargado (columnas de entrada, tipo de salida)
- `POST /predict` - Endpoint de predicción que acepta JSON con las entradas del modelo

### Ejemplo de request para `/predict`

```json
{
  "feature_1": 1.0,
  "feature_2": 2.0,
  "feature_3": 3.0
}
```

### Ejemplo de response

```json
{
  "prediction": 0.85,
  "input": {
    "feature_1": 1.0,
    "feature_2": 2.0,
    "feature_3": 3.0
  }
}
```

## Características

- **Inferencia dinámica**: Detecta automáticamente el framework ML desde `requirements.txt`
- **Detección de columnas**: Infiere las columnas de entrada desde el modelo y datos de referencia
- **Validación automática**: Valida que todas las columnas requeridas estén presentes en la request
- **Idempotencia**: Los scripts pueden ejecutarse múltiples veces sin efectos secundarios
- **Recursos mínimos**: La app usa 0.25 CPU y 0.5 GB RAM por defecto

## Frameworks soportados

- scikit-learn / sklearn
- xgboost
- lightgbm
- catboost
- tensorflow
- pytorch

## Notas

- Los scripts son idempotentes: puedes ejecutarlos múltiples veces
- `deploy.sh` actualiza la app si ya existe
- El modelo se carga al iniciar la aplicación
- Las columnas de entrada se infieren desde el modelo y `data.parquet`
