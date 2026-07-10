.PHONY: install dev-install test eval lint docker-build docker-run clean

# ── Install ──────────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# ── Test & Eval ───────────────────────────────────────────────────────────────
test:
	pytest

eval:
	python -m llm_playbook.eval.retrieval_eval \
		--dataset eval/qa_set.jsonl \
		--k 3 5

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ examples/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t llm-playbook:latest .

docker-run:
	docker run --env-file .env llm-playbook:latest \
		python examples/01_min_rag_cli.py --query "$(QUERY)"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist *.egg-info
