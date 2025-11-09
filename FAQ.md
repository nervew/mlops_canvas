# FAQ.md

**¿Necesito cuenta Azure?** Principiante: no, ejecuta local. Experta: sí, para redes privadas.

**¿Dónde configuro rutas?** `configs/<env>/config.yaml` para datos y umbrales.【F:configs/dev/config.yaml†L1-L6】

**¿Cómo cambio el modelo?** Modifica `src/training/train.py` y `params.yaml`, reentrena y registra via pipeline.【F:src/training/train.py†L18-L65】【F:src/training/params.yaml†L1-L5】

**¿Se soporta blue/green?** Sí, ajustando Helm y workflows `cd-qa.yml`/`cd-prod.yml`.

**¿Qué hago si falla batch?** Verifica `batch_job.py` y `schema_contract.py` para alinear esquemas.【F:src/inference/batch/batch_job.py†L1-L56】【F:src/inference/batch/schema_contract.py†L1-L15】
