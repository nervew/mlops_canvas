# RELEASES.md

**Versionado:** SemVer (`MAJOR.MINOR.PATCH`); MINOR para features compatibles, PATCH para fixes.

**Flujo:** 1) actualiza `CHANGELOG.md` y `mkdocs.yml`; 2) `git tag -a vX.Y.Z -m "Release"`; 3) `make release ENV=qa`; 4) promueve con `cd-prod.yml` si error rate <2% y smoke OK; 5) publica GitHub Release con métricas.

**Rollback:** `helm rollback online-service <revision>`, `az ml model set-default` a versión previa, `terraform ... apply -refresh-only` para validar estado.

**Checklist:** imágenes ACR versionadas, pipelines AML completos, dashboards/alertas actualizados.

**Rutas:** Principiante → ensaya tags `-rc` en `dev`; Experta → automatiza changelog y firma tags.
