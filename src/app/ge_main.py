# IMPORTS Y CONFIGURACIÓN INICIAL
import pandas as pd

# Supongamos que las clases FeatureValidatorTrain y FeatureValidatorCheck
# están definidas en un módulo llamado feature_validator.py
# from feature_validator import FeatureValidatorTrain, FeatureValidatorCheck

# Para este ejemplo, las clases ya están definidas en el notebook o importadas

# --- PASO 1: Validación Inicial (Entrenamiento) ---

# Cargar o crear tu dataset original (features + target opcional)
df_train = pd.DataFrame({
    "feature_1": [1, 2, 3, 4, 5],
    "feature_2": [10, 20, 30, 40, 50],
    "target": [0, 1, 0, 1, 0]
})

# Crear instancia del validador para entrenamiento
validator_train = FeatureValidatorTrain(
    expectation_suite_name="feature_validation_suite",
    expectations_path="/dbfs/great_expectations/"
)

# Generar y guardar expectativas basadas en el dataset original
initial_report = validator_train.fit(df_train)

# Mostrar resumen del reporte inicial
print("Reporte inicial de validacion:")
print(f"Success: {initial_report['success']}")
print(f"Statistics: {initial_report['statistics']}")

# --- PASO 2: Validación en Pipeline (Evaluación) ---

# Dataset nuevo a validar (puede ser otro dataset real de pipeline)
df_new = pd.DataFrame({
    "feature_1": [2, 3, 4, None, 6],  # Aquí hay un valor faltante intencional
    "feature_2": [15, 25, 35, 45, 55],
    "target": [1, 0, 1, 0, 1]
})

# Crear instancia del validador para validación
validator_check = FeatureValidatorCheck(
    expectation_suite_name="feature_validation_suite",
    expectations_path="/dbfs/great_expectations/"
)

# Validar el nuevo dataset con las reglas guardadas
is_valid, validation_report = validator_check.validate(df_new)

print("Validación en pipeline:")
print(f"¿Datos correctos?: {is_valid}")

# Si quieres revisar detalles de la validación (ej. fallos)
failures = [r for r in validation_report["results"] if not r["success"]]
print(f"Número de fallos: {len(failures)}")
if failures:
    for f in failures:
        print(f"Expectation: {f['expectation_config']['expectation_type']}")
        print(f"Mensaje: {f.get('result', {}).get('unexpected_list', [])}\n")
