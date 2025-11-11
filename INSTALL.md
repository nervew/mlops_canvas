# INSTALL.md

## Prerrequisitos
SO: Ubuntu 20.04+, macOS 12+, Windows 11 WSL2 (amd64). Python 3.10+, Docker 24+, Azure CLI 2.45+ (`az extension add -n ml`). Terraform 1.4+ (asumido) con rol `Contributor`. Acceso a GitHub Actions.

## Matriz
| SO | CPU | Herramientas |
|----|-----|--------------|
| Ubuntu 20.04+ | x86_64 | Python, Docker, Azure CLI, Terraform |
| macOS 12+ | ARM/x86_64 | Python, Docker Desktop, Azure CLI |
| Windows 11 WSL2 | x86_64 | Ubuntu WSL, Docker Desktop, Azure CLI |

## Instalación online
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
az extension add -n ml -y || az extension update -n ml
```

## Instalación offline
1. `pip wheel -r requirements.txt -w wheelhouse` (con red).
2. Copia `wheelhouse/` y `requirements.txt` al entorno aislado.
3. `pip install --no-index --find-links=wheelhouse -r requirements.txt`.
4. Construye imágenes Docker con dependencias vendorizadas.

## Verificación
`python --version` ≥3.10, `pytest -q` sin fallos, `terraform -chdir=infra/terraform init` exitoso.

## Variables sensibles
Usa `.env.example` (ver [OPERATIONS](OPERATIONS.md)); almacena secretos en Key Vault o GitHub Secrets.

## Errores comunes
Proxy corporativo → `az config set core.use_adal_cache=true`. Docker sin permisos → agrega usuario al grupo `docker`. Terraform bloqueado → revisa backend.
