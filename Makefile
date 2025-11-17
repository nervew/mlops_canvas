.PHONY: test lint build push deploy

test:
pytest -q

build:
docker build -t training-api ./training_api
docker build -t inference-api ./inference_api

push:
./infra/azure/create_acr_and_push.sh

deploy:
./infra/azure/deploy_functions.sh
