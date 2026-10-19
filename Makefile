install:
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python download_data.py

eda:
	python -m src.eda

train:
	python -m src.train

test:
	pytest -q

lint:
	ruff check .

run:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

all: data eda train test lint
