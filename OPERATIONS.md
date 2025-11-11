# OPERATIONS.md

**Rutina diaria:** revisar dashboards App Insights (latencia/errores/éxito), validar jobs batch y particiones ADLS, confirmar pipelines AML.

**Despliegues:** `make plan ENV=<env>` → aprobar `cd-<env>.yml` → verificar smoke (`tests/e2e/test_smoke.py`).【F:tests/e2e/test_smoke.py†L1-L12】 Actualiza `RELEASES.md`.

**Config:** variables `AML_WORKSPACE`, `AML_RESOURCE_GROUP`, `ACR_NAME`, `APPINSIGHTS_CONNECTION_STRING` en `.env.example` y Key Vault.

**Backups:** artefactos AML/MLflow versionados en Storage, dashboards/alertas como código, snapshots del estado Terraform.

**Observabilidad:** `telemetry.py` envía trazas a App Insights,【F:src/monitoring/telemetry.py†L1-L22】 `/metrics` para Prometheus, alerta si error rate >2% por 5 min.

**Rutas:** Principiante → practicar en `dev`; Experta → documentar runbooks y pruebas de restauración trimestrales.
