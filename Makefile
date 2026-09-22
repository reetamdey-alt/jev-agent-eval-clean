.PHONY: install lint typecheck test smoke clean

install:
	uv sync

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run mypy src/jev_agent_eval

test:
	uv run pytest -q

# Offline smoke eval against bundled fixtures using the mock provider.
smoke:
	uv run jev-eval run --dataset datasets/manifests/core-v1.yaml --model mock-jev --suite smoke --provider mock

clean:
	rm -rf reports/runs reports/cache .pytest_cache .ruff_cache .mypy_cache
