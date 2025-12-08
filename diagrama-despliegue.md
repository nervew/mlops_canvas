# Diagrama de Despliegue y Componentes - MLOps Canvas

## Diagrama de Componentes del Sistema

```mermaid
graph TB
    subgraph "Cliente"
        CLIENT[Cliente HTTP/REST]
    end
    
    subgraph "API FastAPI"
        API[app.py<br/>FastAPI Application]
        CONFIG[config.py<br/>Configuración]
        PREDICT[Endpoint /predict]
        HEALTH[Endpoint /health]
        INFO[Endpoint /model-info]
        ROOT[Endpoint /]
    end
    
    subgraph "Capa de Modelo ML"
        LOADER[Cargador de Modelo<br/>load_model]
        FRAMEWORK[Detector de Framework<br/>detect_framework]
        INFER[Inferencia de Columnas<br/>infer_input_columns]
        OUTPUT[Inferencia de Salida<br/>infer_output_type]
    end
    
    subgraph "Artifacts"
        MODEL[model.pkl<br/>Modelo Serializado]
        DATA[data.parquet<br/>Datos de Referencia]
        REQ[requirements.txt<br/>Dependencias]
    end
    
    subgraph "Lógica de Negocio"
        EXEC[_execute_prediction<br/>Ejecución de Predicción]
        HANDLER[handle_api_error<br/>Manejo de Errores]
        SAFE[safe_api_call<br/>Llamadas Seguras]
    end
    
    CLIENT --> API
    API --> CONFIG
    API --> PREDICT
    API --> HEALTH
    API --> INFO
    API --> ROOT
    
    PREDICT --> EXEC
    EXEC --> HANDLER
    HANDLER --> SAFE
    SAFE --> LOADER
    
    LOADER --> MODEL
    LOADER --> FRAMEWORK
    FRAMEWORK --> REQ
    
    INFER --> DATA
    INFER --> MODEL
    OUTPUT --> DATA
    OUTPUT --> MODEL
    
    EXEC --> INFER
    EXEC --> OUTPUT
```

## Diagrama de Despliegue en Azure

```mermaid
graph TB
    subgraph "Desarrollo Local"
        DEV[Desarrollador]
        DOCKER_LOCAL[Docker Local<br/>docker-local.sh]
        CONTAINER_LOCAL[Contenedor Local<br/>localhost:8000]
    end
    
    subgraph "Azure Cloud"
        subgraph "Azure Subscription"
            SUB[Subscription<br/>Gobierno de datos]
            
            subgraph "Resource Group"
                RG[GRPANALITICA]
                
                subgraph "Azure Container Registry"
                    ACR[ACR: mlopstestacr<br/>SKU: Basic]
                    IMAGE[Imagen Docker<br/>mlopstest4-api:0.0.3-dev.1]
                end
                
                subgraph "Container Apps Environment"
                    ENV[Environment<br/>mlopstestenvironment4]
                    
                    subgraph "Container App"
                        APP[Container App<br/>mlopstestapp4]
                        REPLICAS[Réplicas Auto-escaladas]
                        INGRESS[Ingress External<br/>HTTPS]
                    end
                end
            end
        end
        
        subgraph "Azure CLI"
            AZ_CLI[Azure CLI]
            SETUP[setup.sh<br/>Configuración Inicial]
            DEPLOY[deploy.sh<br/>Despliegue Continuo]
        end
    end
    
    subgraph "Artefactos de Build"
        DOCKERFILE[Dockerfile]
        APP_PY[app.py]
        CONFIG_PY[config.py]
        ARTIFACTS_FOLDER[artifacts/<br/>model.pkl<br/>data.parquet<br/>requirements.txt]
    end
    
    subgraph "Clientes Externos"
        USERS[Usuarios/Clientes]
        API_REQUESTS[Requests HTTP/HTTPS]
    end
    
    DEV --> DOCKER_LOCAL
    DOCKER_LOCAL --> CONTAINER_LOCAL
    
    DEV --> SETUP
    DEV --> DEPLOY
    
    SETUP --> AZ_CLI
    DEPLOY --> AZ_CLI
    
    AZ_CLI --> SUB
    AZ_CLI --> RG
    
    DOCKERFILE --> IMAGE
    APP_PY --> IMAGE
    CONFIG_PY --> IMAGE
    ARTIFACTS_FOLDER --> IMAGE
    
    DEPLOY --> ACR
    ACR --> IMAGE
    
    IMAGE --> APP
    APP --> REPLICAS
    APP --> ENGRESS
    ENV --> APP
    
    USERS --> API_REQUESTS
    API_REQUESTS --> INGRESS
    INGRESS --> REPLICAS
    REPLICAS --> APP
```

## Flujo de Despliegue

```mermaid
sequenceDiagram
    participant Dev as Desarrollador
    participant Docker as Docker Local
    participant ACR as Azure Container Registry
    participant CLI as Azure CLI
    participant Env as Container App Environment
    participant App as Container App
    participant Users as Usuarios
    
    Note over Dev,Users: Desarrollo Local
    Dev->>Docker: docker-local.sh
    Docker->>Docker: Build imagen
    Docker->>Docker: Run contenedor
    Docker-->>Dev: API en localhost:8000
    
    Note over Dev,Users: Setup Inicial (una vez)
    Dev->>CLI: setup.sh
    CLI->>CLI: Crear Resource Group
    CLI->>CLI: Crear ACR
    CLI->>CLI: Crear Environment
    
    Note over Dev,Users: Despliegue a Producción
    Dev->>CLI: deploy.sh
    CLI->>Docker: Build imagen
    Docker->>Docker: Tag imagen
    Docker->>ACR: Push imagen
    CLI->>ACR: Login y autenticación
    CLI->>App: Crear/Actualizar Container App
    App->>Env: Asociar a Environment
    App->>ACR: Pull imagen
    App->>App: Iniciar réplicas
    App-->>CLI: URL pública asignada
    
    Note over Dev,Users: Uso en Producción
    Users->>App: Request HTTP/HTTPS
    App->>App: Cargar modelo
    App->>App: Ejecutar predicción
    App-->>Users: Response JSON
```

## Arquitectura de Componentes Detallada

```mermaid
graph LR
    subgraph "Contenedor Docker"
        subgraph "Aplicación Python"
            UVICORN[Uvicorn ASGI Server]
            FASTAPI[FastAPI App]
            
            subgraph "Módulos"
                APP_MOD[app.py]
                CONFIG_MOD[config.py]
            end
            
            subgraph "Inicialización"
                STARTUP[Startup Event]
                LOAD_MODEL[Cargar Modelo]
                INFER_COLS[Inferir Columnas]
                CREATE_SCHEMA[Crear Schema Pydantic]
            end
        end
        
        subgraph "Artifacts Montados"
            MODEL_FILE[model.pkl]
            DATA_FILE[data.parquet]
            REQ_FILE[requirements.txt]
        end
        
        subgraph "Dependencias Python"
            SKLEARN[scikit-learn]
            XGBOOST[xgboost]
            LIGHTGBM[lightgbm]
            PANDAS[pandas]
            NUMPY[numpy]
        end
    end
    
    subgraph "Endpoints REST"
        EP1[GET /]
        EP2[GET /health]
        EP3[GET /model-info]
        EP4[POST /predict]
    end
    
    UVICORN --> FASTAPI
    FASTAPI --> APP_MOD
    APP_MOD --> CONFIG_MOD
    FASTAPI --> STARTUP
    STARTUP --> LOAD_MODEL
    LOAD_MODEL --> MODEL_FILE
    LOAD_MODEL --> REQ_FILE
    STARTUP --> INFER_COLS
    INFER_COLS --> DATA_FILE
    STARTUP --> CREATE_SCHEMA
    
    FASTAPI --> EP1
    FASTAPI --> EP2
    FASTAPI --> EP3
    FASTAPI --> EP4
    
    EP4 --> LOAD_MODEL
    LOAD_MODEL --> SKLEARN
    LOAD_MODEL --> XGBOOST
    LOAD_MODEL --> LIGHTGBM
    INFER_COLS --> PANDAS
    INFER_COLS --> NUMPY
```

## Especificaciones de Despliegue

### Configuración del Container App
- **CPU**: 0.25 cores
- **Memoria**: 0.5 GiB
- **Puerto**: 8000
- **Ingress**: External (público)
- **Escalado**: Automático según demanda

### Componentes del Build
- **Base Image**: python:3.11-slim
- **Puerto Exposición**: 8000
- **Command**: uvicorn app:app --host 0.0.0.0 --port 8000

### Recursos Azure
- **Resource Group**: GRPANALITICA
- **ACR**: mlopstestacr (Basic SKU)
- **Environment**: mlopstestenvironment4
- **App Name**: mlopstestapp4
- **Location**: eastus

## Arquitectura de Componentes Azure - Detallada

```mermaid
graph TB
    subgraph "Azure Subscription"
        SUB["<b>Azure Subscription</b><br/>Nombre: Gobierno de datos<br/>Rol: Contenedor de recursos<br/>Facturación y acceso"]
        
        subgraph "Resource Group: GRPANALITICA"
            RG["<b>Resource Group</b><br/>Grupo lógico de recursos<br/>Location: eastus<br/>Gestión y organización"]
            
            subgraph "Azure Container Registry"
                ACR["<b>Azure Container Registry (ACR)</b><br/>Nombre: mlopstestacr<br/>SKU: Basic<br/>Rol: Repositorio de imágenes Docker"]
                
                ACR_REPO["<b>Repositorios</b><br/>Almacenamiento de imágenes<br/>Versionado por tags"]
                
                ACR_AUTH["<b>Autenticación</b><br/>Admin credentials habilitado<br/>Username/Password para pull"]
                
                ACR_IMAGES["<b>Imágenes Almacenadas</b><br/>mlopstest4-api:0.0.3-dev.1<br/>Historial de versiones"]
            end
            
            subgraph "Container Apps Environment"
                ENV["<b>Container Apps Environment</b><br/>Nombre: mlopstestenvironment4<br/>Rol: Entorno aislado para apps<br/>Red compartida y configuración"]
                
                ENV_NET["<b>Networking</b><br/>Red virtual compartida<br/>Aislamiento lógico"]
                
                ENV_LOGS["<b>Log Analytics</b><br/>Centralización de logs<br/>Monitoreo y diagnóstico"]
                
                subgraph "Container App: mlopstestapp4"
                    APP["<b>Container App</b><br/>Aplicación principal<br/>Orquestación de contenedores"]
                    
                    APP_REVIS["<b>Revisiones</b><br/>Versiones de despliegue<br/>Rollback y versionado"]
                    
                    APP_REPLICAS["<b>Réplicas</b><br/>CPU: 0.25 cores<br/>Memoria: 0.5 GiB<br/>Auto-escalado 0-N"]
                    
                    APP_INGRESS["<b>Ingress Controller</b><br/>Tipo: External<br/>Protocolo: HTTPS<br/>Balanceador de carga"]
                    
                    APP_DNS["<b>DNS/FQDN</b><br/>URL pública automática<br/>SSL/TLS automático<br/>Certificado gestionado"]
                    
                    subgraph "Contenedores en Ejecución"
                        CONTAINER1["<b>Contenedor 1</b><br/>Python 3.11-slim<br/>FastAPI + Uvicorn<br/>Puerto 8000"]
                        
                        CONTAINER2["<b>Contenedor 2</b><br/>Python 3.11-slim<br/>FastAPI + Uvicorn<br/>Puerto 8000"]
                        
                        CONTAINER_N["<b>Contenedor N</b><br/>Escalado horizontal<br/>Según demanda"]
                    end
                end
            end
            
            subgraph "Azure CLI y Gestión"
                AZ_CLI["<b>Azure CLI</b><br/>Interfaz de línea de comandos<br/>Gestión de recursos"]
                
                SETUP_SCRIPT["<b>setup.sh</b><br/>Configuración inicial<br/>Creación de recursos"]
                
                DEPLOY_SCRIPT["<b>deploy.sh</b><br/>Ciclo de despliegue<br/>CI/CD pipeline"]
            end
        end
    end
    
    subgraph "Red Pública"
        INTERNET[Internet]
        USERS[Usuarios/Clientes]
        HTTPS_REQUESTS[Requests HTTPS]
    end
    
    subgraph "Desarrollo Local"
        DEV[Desarrollador]
        DOCKER_LOCAL[Docker Local]
        DOCKER_BUILD[Build Local]
    end
    
    SUB --> RG
    RG --> ACR
    RG --> ENV
    ENV --> APP
    
    ACR --> ACR_REPO
    ACR --> ACR_AUTH
    ACR_REPO --> ACR_IMAGES
    
    ENV --> ENV_NET
    ENV --> ENV_LOGS
    
    APP --> APP_REVIS
    APP --> APP_REPLICAS
    APP --> APP_INGRESS
    APP_INGRESS --> APP_DNS
    APP_REPLICAS --> CONTAINER1
    APP_REPLICAS --> CONTAINER2
    APP_REPLICAS --> CONTAINER_N
    
    DEV --> DOCKER_LOCAL
    DOCKER_LOCAL --> DOCKER_BUILD
    DOCKER_BUILD -->|Push| ACR
    
    AZ_CLI --> SETUP_SCRIPT
    AZ_CLI --> DEPLOY_SCRIPT
    SETUP_SCRIPT --> RG
    DEPLOY_SCRIPT --> ACR
    DEPLOY_SCRIPT --> APP
    
    APP -->|Pull| ACR
    APP_INGRESS -->|Autenticación| ACR_AUTH
    
    INTERNET --> HTTPS_REQUESTS
    USERS --> HTTPS_REQUESTS
    HTTPS_REQUESTS --> APP_DNS
    APP_DNS --> APP_INGRESS
    APP_INGRESS --> CONTAINER1
    APP_INGRESS --> CONTAINER2
    APP_INGRESS --> CONTAINER_N
    
    ENV_LOGS --> CONTAINER1
    ENV_LOGS --> CONTAINER2
    ENV_LOGS --> CONTAINER_N
```

## Flujo de Componentes Azure

```mermaid
flowchart LR
    subgraph "1. Infraestructura Base"
        SUB1[Azure Subscription]
        RG1[Resource Group]
        SUB1 --> RG1
    end
    
    subgraph "2. Repositorio de Imágenes"
        ACR1[ACR]
        ACR_REPO1[Repositorio]
        ACR_AUTH1[Auth]
        ACR1 --> ACR_REPO1
        ACR1 --> ACR_AUTH1
    end
    
    subgraph "3. Entorno de Ejecución"
        ENV1[Container Apps Environment]
        ENV_NET1[Networking]
        ENV_LOG1[Log Analytics]
        ENV1 --> ENV_NET1
        ENV1 --> ENV_LOG1
    end
    
    subgraph "4. Aplicación"
        APP1[Container App]
        APP_REV1[Revisiones]
        APP_REP1[Réplicas]
        APP_ING1[Ingress]
        APP1 --> APP_REV1
        APP1 --> APP_REP1
        APP1 --> APP_ING1
    end
    
    subgraph "5. Contenedores"
        CONT1[Contenedor<br/>FastAPI]
        CONT2[Contenedor<br/>FastAPI]
        CONTN[Más contenedores<br/>auto-escalados]
    end
    
    RG1 --> ACR1
    RG1 --> ENV1
    ENV1 --> APP1
    APP_REP1 --> CONT1
    APP_REP1 --> CONT2
    APP_REP1 --> CONTN
    
    ACR1 -.Imagen.-> APP1
    APP_ING1 -.Tráfico.-> CONT1
    APP_ING1 -.Tráfico.-> CONT2
    APP_ING1 -.Tráfico.-> CONTN
    CONT1 -.Logs.-> ENV_LOG1
    CONT2 -.Logs.-> ENV_LOG1
    CONTN -.Logs.-> ENV_LOG1
```

## Componentes Azure - Descripción de Roles

```mermaid
mindmap
    root((Componentes<br/>Azure))
        Subscription
            Contenedor de recursos
            Facturación centralizada
            Control de acceso (IAM)
            Límites y cuotas
        Resource Group
            Agrupación lógica
            Lifecycle management
            Tagging y organización
            Permisos y políticas
        Azure Container Registry
            Repositorio Docker
            Versionado de imágenes
            Autenticación y seguridad
            Integración CI/CD
            Escaneo de vulnerabilidades
        Container Apps Environment
            Entorno compartido
            Red virtual común
            Log Analytics integrado
            Aislamiento lógico
            Networking unificado
        Container App
            Orquestación de contenedores
            Auto-escalado
            Revisiones y versionado
            Health checks
            Configuración declarativa
        Ingress Controller
            Balanceador de carga
            SSL/TLS automático
            DNS y FQDN
            Routing y reglas
            Métricas y monitoreo
        Log Analytics
            Centralización de logs
            Consultas KQL
            Alertas y dashboards
            Integración con otros servicios
```

## Interacciones entre Componentes Azure

```mermaid
sequenceDiagram
    participant Dev as Desarrollador
    participant CLI as Azure CLI
    participant RG as Resource Group
    participant ACR as Azure Container Registry
    participant Env as Container Apps Environment
    participant App as Container App
    participant Ingress as Ingress Controller
    participant Logs as Log Analytics
    participant Users as Usuarios
    
    Note over Dev,Users: Fase 1: Setup Inicial
    Dev->>CLI: az account set --subscription
    CLI->>RG: Crear Resource Group (setup.sh)
    CLI->>ACR: az acr create (setup.sh)
    CLI->>Env: az containerapp env create (setup.sh)
    
    Note over Dev,Users: Fase 2: Build y Push
    Dev->>CLI: deploy.sh
    CLI->>ACR: az acr login
    CLI->>ACR: docker build
    CLI->>ACR: docker push imagen
    ACR-->>CLI: Imagen almacenada
    
    Note over Dev,Users: Fase 3: Despliegue
    CLI->>App: az containerapp create/update
    App->>Env: Asociar a Environment
    App->>ACR: Autenticar y pull imagen
    ACR-->>App: Imagen descargada
    App->>App: Crear revision
    App->>App: Iniciar réplicas
    App->>Ingress: Configurar routing
    Ingress->>Ingress: Generar FQDN y SSL
    App-->>CLI: URL pública disponible
    
    Note over Dev,Users: Fase 4: Operación
    Users->>Ingress: Request HTTPS
    Ingress->>App: Route a réplica disponible
    App->>App: Procesar request
    App->>Logs: Enviar logs
    App-->>Ingress: Response
    Ingress-->>Users: Response HTTPS
    
    Note over Dev,Users: Monitoreo Continuo
    App->>Logs: Stream logs continuo
    Logs->>Logs: Agregar y analizar
    Logs-->>Dev: Dashboard y alertas
```

## Matriz de Componentes Azure y Responsabilidades

| Componente Azure | Responsabilidad Principal | Características Clave | Integración |
|-----------------|--------------------------|----------------------|-------------|
| **Azure Subscription** | Contenedor de recursos, facturación | Control de acceso, límites | IAM, Cost Management |
| **Resource Group** | Organización y gestión de recursos | Lifecycle, tagging | Todos los recursos |
| **Azure Container Registry** | Repositorio de imágenes Docker | Versionado, autenticación | Container App, CI/CD |
| **Container Apps Environment** | Entorno compartido y aislado | Networking, logs centralizados | Container Apps |
| **Container App** | Orquestación y ejecución | Auto-escalado, revisiones | Environment, ACR |
| **Ingress Controller** | Balanceo de carga y routing | SSL/TLS, DNS automático | Container App |
| **Log Analytics** | Centralización y análisis | KQL queries, alertas | Environment |
