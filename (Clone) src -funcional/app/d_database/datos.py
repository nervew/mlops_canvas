import pandas as pd
import numpy as np

def generate_synthetic_patient_data() -> pd.DataFrame:
    np.random.seed(42)
    hoy = pd.Timestamp.today().normalize()
    fechas = pd.date_range(end=hoy, periods=336, freq='W')
    
    # Sumas con más dispersión agregando ruido gaussiano pequeño
    eps_a = np.random.poisson(lam=25, size=336) + np.random.randint(-2, 3, size=336)
    eps_b = np.random.poisson(lam=30, size=336) + np.random.randint(-2, 3, size=336)
    eps_c = np.random.poisson(lam=20, size=336) + np.random.randint(-2, 3, size=336)
    eps_d = np.random.poisson(lam=15, size=336) + np.random.randint(-2, 3, size=336)

    # Asegura que no hay negativos
    eps_a = np.clip(eps_a, 0, None)
    eps_b = np.clip(eps_b, 0, None)
    eps_c = np.clip(eps_c, 0, None)
    eps_d = np.clip(eps_d, 0, None)

    nuevos_pacientes_totales = eps_a + eps_b + eps_c + eps_d

    df = pd.DataFrame({
        'Semana': fechas,
        'target': nuevos_pacientes_totales,
        'EPS_A': eps_a,
        'EPS_B': eps_b,
        'EPS_C': eps_c,
        'EPS_D': eps_d,
    })
    # Convertir columna 'Semana' a datetime por seguridad
    df["Semana"] = pd.to_datetime(df["Semana"])
    # Renombrar columna de target
    df = df.rename(columns={"Nuevos_pacientes_totales": "target"})

    return df
