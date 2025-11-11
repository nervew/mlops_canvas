# ADR-0002 Uso de Terraform como IaC principal

## Estado
Aceptado (2024-04-30)

## Contexto
Se requiere aprovisionar VNet, AML, AKS, ACR y Key Vault de forma reproducible en `dev/qa/prod`, integrado con CI/CD y redes privadas.

## Decisión
Usar Terraform 1.4+ con módulos internos (`network`, `aml`), `tfvars` por entorno, backend remoto en Azure Storage e integración GitHub Actions.

## Alternativas
- **Bicep**: gran integración Azure pero menor reutilización multi-cloud.
- **Azure CLI scripts**: rápidos pero frágiles para idempotencia/pruebas.

## Consecuencias
+ Versionado Git, planes reproducibles, políticas corporativas.
− Curva de aprendizaje y gestión de estado remoto.

## Seguimiento
Revisar cada 6 meses si Bicep/Terraform CDK aportan valor y mantener `terraform validate`/`fmt` en CI (`ci.yml`).
