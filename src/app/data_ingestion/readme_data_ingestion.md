# data_ingestion
El módulo data_ingestion abstrae la obtención y estandarización inicial de datos externos, sirviendo como punto de entrada del pipeline. Cumple funciones de acceso, carga y formateo de los datos para asegurar que el flujo posterior del pipeline parte de una base consistente y controlada.
```
data_ingestion/
│
├── application/
│   └── service.py          
│       # Orquestación del proceso de ingesta
│       # Recibe: Nada explícitamente (invoca loader)
│       # Retorna: Dataset (objeto de dominio con pd.DataFrame)
│
├── domain/
│   └── dataset.py          
│       # Entidad Dataset (tipado, validación mínima)
│       # Recibe: pd.DataFrame
│       # Retorna: Objeto Dataset
│       # Regla: Solo define estructura de datos, sin lógica de negocio, ni dependencias externas.
│
└── infrastructure/
    └── openml_loader.py    
        # Adaptador concreto para obtener datos desde OpenML
        # Recibe: Nada explícitamente (invoca OpenML API)
        # Retorna: pd.DataFrame limpio (sin credit_amount, target renombrado)
```
## Ejemplo de Uso
```
from data_ingestion.application.service import ingest

# Devuelve un objeto Dataset, con los datos listos para EDA y validación
dataset = ingest()
df = dataset.data
```

