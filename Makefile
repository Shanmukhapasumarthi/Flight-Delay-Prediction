.PHONY: help install pipeline data features train api dashboard test clean lint

help:
	@echo "install    install dependencies"
	@echo "pipeline   run every stage end to end (~15 min)"
	@echo "data       collection -> cleaning -> integration -> features"
	@echo "train      feature selection -> model comparison -> tuning"
	@echo "api        serve the prediction API on :8000"
	@echo "dashboard  serve the Streamlit dashboard on :8501"
	@echo "test       run the test suite"
	@echo "clean      remove generated data and artefacts"

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py

data:
	python run_pipeline.py --from collect --to features

train:
	python run_pipeline.py --from select --to insights

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/*.csv data/interim/*.parquet data/processed/*.parquet
	rm -rf models/*.joblib models/feature_store.json reports/_model_runs
	find . -type d -name __pycache__ -exec rm -rf {} +
