# src/app/d_database/datos.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1_noise(n: int, sigma: float = 250.0, rho: float = 0.55, shocks_prob: float = 0.03, shock_scale: float = 700.0) -> np.ndarray:
    """Ruido AR(1) con shocks ocasionales para evitar sobreajuste."""
    eps = np.zeros(n, dtype=float)
    base = np.random.normal(0.0, sigma, size=n)
    shocks = np.random.rand(n) < shocks_prob
    base[shocks] += np.random.normal(0.0, shock_scale, size=shocks.sum())
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + base[t]
    return eps


def _softmax_rows(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - arr.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def generate_synthetic_patient_data(periods: int = 336) -> pd.DataFrame:
    """
    Genera un dataset semanal con columnas parecidas a las de ADRES/MIPRES.
    La variable objetivo se llama 'y_usuarios_nuevos_semana' (el pipeline la renombra a 'target').
    """
    rng = np.random.default_rng(42)

    # Calendario semanal ~6.5 años; el pipeline filtrará >= 2022
    end = pd.Timestamp.today().normalize()
    fechas = pd.date_range(end=end, periods=periods, freq="W-MON")
    n = len(fechas)

    # Señal latente de demanda (base + estacionalidad anual + ruido AR1 + drift leve)
    base_level = 7200.0
    season = 420.0 * np.sin(2 * np.pi * np.arange(n) / 52.0)
    drift = np.linspace(0, 350, n)  # tendencia suave
    noise = _ar1_noise(n, sigma=260.0, rho=0.55, shocks_prob=0.04, shock_scale=800.0)
    demanda = base_level + season + drift + noise

    # Shares por TIPO (5 categorías → suman 1)
    tipo_latent = np.stack([
        0.6 + 0.15 * np.sin(2 * np.pi * (np.arange(n) +  0) / 52.0) + rng.normal(0, 0.05, n),  # dispositivo
        0.5 + 0.12 * np.sin(2 * np.pi * (np.arange(n) + 13) / 52.0) + rng.normal(0, 0.05, n),  # procedimiento
        0.4 + 0.10 * np.sin(2 * np.pi * (np.arange(n) + 26) / 52.0) + rng.normal(0, 0.05, n),  # medicamento
        0.3 + 0.08 * np.sin(2 * np.pi * (np.arange(n) + 39) / 52.0) + rng.normal(0, 0.05, n),  # nutricional
        0.2 + 0.06 * np.sin(2 * np.pi * (np.arange(n) +  7) / 26.0) + rng.normal(0, 0.05, n),  # servicio_comp
    ], axis=1)
    tipo_shares = _softmax_rows(tipo_latent)

    share_tipo_dispositivo_t4   = tipo_shares[:, 0]
    share_tipo_procedimiento_t4 = tipo_shares[:, 1]
    share_tipo_medicamento_t4   = tipo_shares[:, 2]
    share_tipo_nutricional_t4   = tipo_shares[:, 3]
    share_tipo_servicio_comp_t4 = tipo_shares[:, 4]

    # Shares por RÉGIMEN (3 categorías → suman 1)
    reg_latent = np.stack([
        0.7 + 0.05 * np.sin(2 * np.pi * (np.arange(n) + 0) / 52.0) + rng.normal(0, 0.02, n),  # contributivo
        0.5 + 0.05 * np.sin(2 * np.pi * (np.arange(n) + 8) / 52.0) + rng.normal(0, 0.02, n),  # subsidiado
        0.2 + 0.02 * rng.normal(0, 1, n),                                                     # otro
    ], axis=1)
    reg_shares = _softmax_rows(reg_latent)

    share_regimen_contributivo_t4 = reg_shares[:, 0]
    share_regimen_subsidiado_t4   = reg_shares[:, 1]
    share_regimen_otro_t4         = reg_shares[:, 2]
    hhi_et_t4 = (reg_shares ** 2).sum(axis=1)

    # Volúmenes
    total_suministros_t4       = demanda * (1.08 + rng.normal(0, 0.03, n)) + rng.normal(0, 300, n)
    total_pacientes_unicos_t4  = demanda * (0.27 + rng.normal(0, 0.01, n)) + rng.normal(0, 120, n)

    # “Preferencias” (ligadas a demanda pero con ruido y pequeñas asincronías)
    lag = 2
    d_lag = np.r_[np.repeat(demanda[0], lag), demanda[:-lag]]
    u_nuevos_pref_eps_t4 = (d_lag - 6500) / 9.0  + rng.normal(0, 180, n)
    u_nuevos_pref_eas_t4 = (d_lag - 6400) / 13.0 + rng.normal(0, 160, n)
    u_nuevos_pref_ccf_t4 = (d_lag - 6600) / 12.0 + rng.normal(0, 170, n)
    u_nuevos_pref_ess_t4 = (d_lag - 6300) / 11.0 + rng.normal(0, 150, n)

    # Target “real” ligada a la demanda + combinación de features + más ruido AR(1)
    ruido_target = _ar1_noise(n, sigma=220.0, rho=0.45, shocks_prob=0.03, shock_scale=600.0)
    y_usuarios_nuevos_semana = (
        0.40 * total_suministros_t4
        + 0.22 * u_nuevos_pref_eps_t4
        + 0.10 * u_nuevos_pref_ess_t4
        + 0.05 * u_nuevos_pref_ccf_t4
        + 1300 * (share_regimen_contributivo_t4 - share_regimen_subsidiado_t4)
        + 1600 * (share_tipo_dispositivo_t4 - share_tipo_medicamento_t4)
        + 0.30 * total_pacientes_unicos_t4
        + 400  * np.sin(2 * np.pi * np.arange(n) / 52.0)
        + ruido_target
    )

    # Escalado para mantener órdenes de magnitud razonables sin saturar
    # (normalizamos para que target quede alrededor de 7k–8k)
    y_min, y_max = y_usuarios_nuevos_semana.min(), y_usuarios_nuevos_semana.max()
    y_usuarios_nuevos_semana = 6500 + (y_usuarios_nuevos_semana - y_min) * (1800 / (y_max - y_min))

    df = pd.DataFrame({
        "semana": fechas,
        "total_suministros_t4": total_suministros_t4,
        "total_pacientes_unicos_t4": total_pacientes_unicos_t4,
        "share_tipo_dispositivo_t4": share_tipo_dispositivo_t4,
        "share_tipo_procedimiento_t4": share_tipo_procedimiento_t4,
        "share_tipo_medicamento_t4":   share_tipo_medicamento_t4,
        "share_tipo_nutricional_t4":   share_tipo_nutricional_t4,
        "share_tipo_servicio_comp_t4": share_tipo_servicio_comp_t4,
        "share_regimen_contributivo_t4": share_regimen_contributivo_t4,
        "share_regimen_subsidiado_t4":   share_regimen_subsidiado_t4,
        "share_regimen_otro_t4":         share_regimen_otro_t4,
        "hhi_et_t4": hhi_et_t4,
        "u_nuevos_pref_eps_t4": u_nuevos_pref_eps_t4,
        "u_nuevos_pref_eas_t4": u_nuevos_pref_eas_t4,
        "u_nuevos_pref_ccf_t4": u_nuevos_pref_ccf_t4,
        "u_nuevos_pref_ess_t4": u_nuevos_pref_ess_t4,
        # Target con el nombre “antiguo” que usa la consulta real:
        "y_usuarios_nuevos_semana": y_usuarios_nuevos_semana,
    })

    # Tipos finales
    df["semana"] = pd.to_datetime(df["semana"])
    float_cols = [c for c in df.columns if c != "semana"]
    df[float_cols] = df[float_cols].astype(float)

    return df
