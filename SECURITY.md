# SECURITY.md

**Amenazas:** FastAPI (`/predict`, `/health`, `/metrics`) detrás de Ingress privado/WAF; datos sensibles en ADLS y Key Vault; actores: clientes autenticados, equipo MLOps, adversarios.

**Prácticas:** Managed Identity para secretos, nada en repos; cifrado TLS 1.2+ y Private Endpoints; probes Helm contra DoS; variables en `.env.example` documentadas en [OPERATIONS](OPERATIONS.md).

**Parches/escaneo:** actualización mensual (`pip install -r requirements.txt --upgrade`), Dependabot (Python/Terraform/GitHub Actions), Trivy vía `ci/actions/scan_docker.yml`.

**Reporte:** `security@example.com` con evidencia y severidad (SLA <72h).

**Rutas:** Principiante → checklist de secretos antes de pipelines; Experta → aplicar Azure Policy para forzar TLS y Private Link.
