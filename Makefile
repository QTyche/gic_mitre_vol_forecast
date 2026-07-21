.PHONY: install test lint format smoke

UV ?= uv
export UV_CACHE_DIR ?= .uv-cache

install:
	$(UV) sync --all-extras

test:
	$(UV) run pytest

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy src tests

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

smoke:
	./scripts/reproduce_smoke.sh
	./scripts/reproduce_qrc_smoke.sh
