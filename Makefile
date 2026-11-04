PYTHON ?= .venv/bin/python

.PHONY: check test lint

check: lint test

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	@for script in launchers/*.sh scripts/setup/*.sh scripts/slurm/*.sh; do bash -n "$$script" || exit; done
