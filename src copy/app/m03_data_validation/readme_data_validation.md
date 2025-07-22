# data_validation
```
data_validation/
│
├── application/
│   └── service.py          
│       # Orquesta el proceso de validación de datos
│       # Recibe: 
│       #   - df: pd.DataFrame (datos a validar)
│       #   - profile_path (opcional, ruta perfil JSON)
│       #   - reporte_path (opcional, ruta reporte JSON)
│       #   - fit_profile: bool (si se recalcula el perfil)
│       # Retorna: 
│       #   - DataValidationReport (objeto de dominio con bool y detalles)
│
├── domain/
│   └── report.py           
│       # Entidad DataValidationReport (estructura de reporte)
│       # Recibe: 
│       #   - valido: bool (si pasó validaciones)
│       #   - detalles: dict (problemas, por columna)
│       # Retorna:
│       #   - Objeto DataValidationReport
│       # Regla: 
│       #   - Solo estructura y contrato de reporte.
│       #   - No contiene lógica de validación ni acceso a datos externos.
│
└── infrastructure/
    └── feature_validator.py 
        # Adaptador para validación de features/tablas (implementa reglas)
        # Recibe:
        #   - df: pd.DataFrame (fit, validate)
        #   - profile_path (para guardar/cargar perfil)
        #   - Otros parámetros de configuración (umbral, max_categorias, etc.)
        # Retorna:
        #   - dict con resultados de validación, estructura compatible con dominio
        # Notas:
        #   - Aquí reside TODA la lógica de validación, perfiles, persistencia y comparación
```        
## Ejemplo de uso
```
import pandas as pd
from data_validation.application.service import run

# DataFrame
df = pd.read_csv("datos.csv")

# 1. Ajustar (fit) y guardar perfil de validación en una primera ejecución
reporte = run(
    df,
    profile_path="output/validation/data_profile.json",
    reporte_path="output/validation/validation_report.json",
    fit_profile=True   # ← Aprende el perfil y lo guarda
)

print(reporte.valido)    # True/False: Si los datos cumplen las reglas aprendidas
print(reporte.detalles)  # Dict: Detalles columna a columna de los problemas

# 2. Validar nuevos datos usando el perfil guardado (en pipeline productivo)
nuevo_df = pd.read_csv("nuevos_datos.csv")
reporte_nuevo = run(
    nuevo_df,
    profile_path="output/validation/data_profile.json",
    reporte_path="output/validation/validation_report_nuevos.json",
    fit_profile=False   # ← Solo valida, NO aprende
)

if reporte_nuevo.valido:
    print("Los datos cumplen todas las reglas del perfil.")
else:
    print("Problemas detectados:")
    for columna, detalle in reporte_nuevo.detalles.items():
        print(f"{columna}: {detalle['problemas']}")

```




