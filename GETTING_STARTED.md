# GETTING_STARTED.md

**Objetivo:** pasar del clonado al primer pipeline e inferencia.

## Principiante
1. `python src/training/data_prep.py --input data/raw --output data/processed`
2. ```bash
python src/training/train.py --data data/processed \
  --parameters src/training/params.yaml --output outputs/model
```
3. ```bash
uvicorn src.inference.online.app:app --host 0.0.0.0 --port 8080
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"features": [0.1,0.2,0.3]}'
```
4. `pytest tests/unit -q`
5. Continúa con despliegue `dev` en [OPERATIONS](OPERATIONS.md).

## Experta
1. `make plan ENV=dev && make apply ENV=dev`
2. `az ml component create --file <component.yaml>`
3. `az ml job create -f mlops/aml/pipelines/train_pipeline.yaml -w <ws> -g <rg>`
4. ```bash
az acr login -n <acr>
docker build -f docker/Dockerfile.online -t <acr>.azurecr.io/online:local .
docker push <acr>.azurecr.io/online:local
```
5. ```bash
helm upgrade --install online-service helm/online-service \
  --set image.repository=<acr>.azurecr.io/online --set image.tag=local
```
6. Integra `.github/workflows`.

## Verificación y siguiente paso
`/health` → `{ "status": "ok" }`, pipeline registra `roc_auc`, `outputs/model` contiene artefactos MLflow; personaliza `configs/<env>/config.yaml`, registra ADR y revisa [TROUBLESHOOTING](TROUBLESHOOTING.md).
