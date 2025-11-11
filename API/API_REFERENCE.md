# API_REFERENCE.md

## Endpoints
| Ruta | Método | Uso | Estados | Auth |
|------|--------|-----|---------|------|
| `/health` | GET | Comprobar servicio | 200 | Red privada |
| `/predict` | POST | Inferencia | 200, 422 | Token Ingress |
| `/metrics` | GET | Prometheus scrape | 200 | Scraper autorizado |

## Ejemplo
```bash
curl -X POST https://dev-online.example.com/predict -H "Content-Type: application/json" \
  -d '{"features": [0.12, 0.87, 0.33]}'
```

**Request** `{"features": [0.12, 0.87, 0.33]}`

**Response** `{"score": 0.74, "label": 1}`

## Errores
- 422: datos fuera de rango o longitud cero.
- 401: autenticación externa fallida.

## Contratos
- Pydantic (`PredictionRequest`, `PredictionResponse`).【F:src/inference/online/schema.py†L1-L14】
- Métricas Prometheus via `CollectorRegistry` + middleware `X-Request-Id`.【F:src/inference/online/app.py†L13-L55】

## Versionado y rutas
`app.version = 1.0.0`; breaking changes → `/v2`. Principiante: verifica `/health`. Experta: integra `/metrics` y documenta cambios en ADR.
