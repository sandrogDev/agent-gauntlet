.PHONY: install lint fmt typecheck test check demo docker clean

PY ?= python3
SUITE := examples/suites/airline_support.yaml
DEMO_AGENT := python:examples/agents/naive_agent.py

install:
	$(PY) -m pip install -e ".[dev,all]"
	pre-commit install || true

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest --cov

check: lint typecheck test

demo:
	gauntlet run $(SUITE) --provider $(DEMO_AGENT) --html gauntlet-report.html

live:
	gauntlet run $(SUITE) --provider anthropic:claude-sonnet-5-5 --baseline latest --html gauntlet-live.html

docker:
	docker build -t agent-gauntlet .
	docker run --rm agent-gauntlet run $(SUITE) --provider $(DEMO_AGENT)

clean:
	rm -rf .gauntlet .pytest_cache .mypy_cache .ruff_cache coverage.xml .coverage *.html dist build
