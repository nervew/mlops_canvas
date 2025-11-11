# TROUBLESHOOTING.md

| Síntoma | Causa | Acción |
|---------|-------|--------|
| `terraform init` falla | Rol insuficiente | Solicita `Contributor` o usa Managed Identity |
| `/predict` devuelve 500 | Modelo no inicializado | Revisar logs y reiniciar `InferenceService` |
| Pipeline AML detenido | Componente no registrado | Validar versiones y registrar de nuevo |
| Batch sin particiones | Error de esquema | Ajustar datos vs `schema_contract.py` |

**Pasos comunes:** revisar logs App Insights/stdout, `pytest -k` para aislar fallos, verificar Private Endpoints con `az network private-endpoint list`.

**Rutas:** Principiante → ejecutar `pytest -q` y `curl /health`; Experta → alertas Log Analytics y runbooks en Azure Automation.
