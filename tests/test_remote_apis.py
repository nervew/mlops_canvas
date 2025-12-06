import os

import pytest
import requests


def _get_base_url(env_var_name: str) -> str:
    """
    Obtiene la URL base de la API desde una variable de entorno.

    Si no está definida, marca el test como skipped para no fallar
    en entornos locales o de CI donde no se tenga la API desplegada.
    """
    base_url = os.getenv(env_var_name)
    if not base_url:
        pytest.skip(f"{env_var_name} no está definida; se omite test de API remota")
    return base_url.rstrip("/")


def test_training_api_status():
    """
    Test de integración sencillo contra la API de entrenamiento desplegada en Azure.

    Requiere la variable de entorno TRAINING_API_BASE_URL, por ejemplo:
      https://train-func.azurewebsites.net
    """
    base_url = _get_base_url("TRAINING_API_BASE_URL")

    resp = requests.get(f"{base_url}/status", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_inference_api_health():
    """
    Test de integración sencillo contra la API de inferencia desplegada en Azure.

    Requiere la variable de entorno INFERENCE_API_BASE_URL, por ejemplo:
      https://predict-func.azurewebsites.net
    """
    base_url = _get_base_url("INFERENCE_API_BASE_URL")

    resp = requests.get(f"{base_url}/healthz", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


