.PHONY: setup test docs docs-lint docs-linkcheck plan apply release
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

docs:
	mkdocs build

docs-lint:
	markdownlint "*.md" "API/*.md" "ADR/*.md"

docs-linkcheck:
	linkchecker http://localhost:8000 || true

plan:
	terraform -chdir=infra/terraform plan -var="environment=${ENV}" -out=tf.plan

apply:
	terraform -chdir=infra/terraform apply tf.plan

release:
	$(MAKE) plan ENV=${ENV}
