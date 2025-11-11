# CONTRIBUTING.md

**Código de conducta:** sigue el [Código de Conducta](CODE_OF_CONDUCT.md).

**Flujo:** crea issue, rama `feat/<tema>` o `fix/<bug>`, usa Conventional Commits, ejecuta `pytest`, `make docs-lint`, `make docs`, abre PR con [plantilla](PULL_REQUEST_TEMPLATE.md), solicita revisión y haz squash tras CI verde.

**Calidad mínima:** cobertura ≥80% (`pytest --cov`), `terraform fmt -check` + `terraform validate`, Dockerfiles pasan `ci/actions/scan_docker.yml`.

**Revisión:** checklists de seguridad/docs/pruebas, cero secretos, registra impactos en `RELEASES.md` y ADR cuando aplique.

**Rutas:** Principiante → completa [GETTING_STARTED](GETTING_STARTED.md) antes de cambios profundos. Experta → prepara rollback y enlaza ADRs.

**DoD docs:** MkDocs sin warnings (`make docs`), enlaces verificados (`make docs-linkcheck`), snippets probados.
