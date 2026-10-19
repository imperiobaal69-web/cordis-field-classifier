.PHONY: setup data train test lint serve image

setup:            ## install core + dev dependencies
	uv sync --group dev

setup-all:        ## + PyTorch stack for embedding / fine-tuning runs
	uv sync --group dev --extra embed --extra finetune --extra report

data:             ## download CORDIS exports and build the labelled dataset
	uv run fieldclf download
	uv run fieldclf build

train:            ## reproduce every leaderboard row (~2 h on Apple M4)
	uv run python scripts/run_experiments.py

figures:          ## regenerate reports/figures and reports/error_analysis.md
	uv run python scripts/make_report.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run mypy src

serve:
	uv run fieldclf serve

image:
	docker build --build-arg MODEL_DIR=artifacts/tfidf_logreg -t fieldclf .
