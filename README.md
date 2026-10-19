# MLOps Assignment 01 — End-to-End Heart Disease ML System

This repository implements the complete workflow specified in **MLOps Assignment 01**:

1. Data acquisition and EDA
2. Feature engineering and model development
3. MLflow experiment tracking
4. Model packaging and reproducibility
5. Pytest + GitHub Actions CI/CD
6. FastAPI + Docker model serving
7. Kubernetes production deployment
8. Prometheus + Grafana monitoring
9. Documentation/reporting

> **Academic-integrity note:** personalize the experiments, plots, thresholds, architecture decisions, screenshots, report wording, and conclusions before submission. The assignment explicitly requires individual effort and warns against highly similar implementations.

## Project structure

```text
.
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── plots/
│   └── metrics/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── k8s/
│   ├── namespace.yaml
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── prometheus-config.yaml
│   ├── prometheus-deployment.yaml
│   ├── grafana-config.yaml
│   ├── grafana-dashboard-configmap.yaml
│   └── grafana-deployment.yaml
├── monitoring/prometheus/prometheus.yml
├── monitoring/grafana/dashboards/ml-api-dashboard.json
├── report.md
└── download_data.py
```

## 1. Local setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 2. Download the UCI dataset

```bash
python download_data.py
```

This retrieves UCI Heart Disease dataset ID 45 using `ucimlrepo`, converts the target to binary (0 = absence, 1 = presence), normalizes missing-value markers, and writes:

```text
data/raw/heart_disease.csv
data/raw/heart_disease_metadata.json
```

## 3. Run EDA

```bash
python -m src.eda
```

Expected outputs include:

```text
artifacts/plots/feature_histograms.png
artifacts/plots/correlation_heatmap.png
artifacts/plots/class_balance.png
artifacts/plots/boxplots_numeric.png
artifacts/metrics/data_summary.json
```

## 4. Train and track experiments

```bash
python -m src.train
```

This:

- creates a leakage-safe preprocessing pipeline;
- evaluates Logistic Regression and Random Forest;
- performs GridSearchCV;
- uses stratified cross-validation;
- calculates accuracy, precision, recall and ROC-AUC;
- logs parameters, CV results, metrics, plots and models to MLflow;
- selects the final model by validation ROC-AUC;
- saves the complete preprocessing + model pipeline.

Start MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000.

## 5. Run tests

```bash
pytest -q
```

Run lint:

```bash
ruff check .
```

## 6. Run the API locally

Training creates `artifacts/model.joblib`.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

The response contains `prediction` and `confidence`.

## 7. Docker

Build:

```bash
docker build -t heart-disease-mlops:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

Test:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Docker Compose:

```bash
docker compose up --build
```

The compose stack starts:

- FastAPI API on port 8000
- MLflow on port 5000
- Prometheus on port 9090
- Grafana on port 3000

Grafana default local login:

```text
username: admin
password: admin
```

Change the password for a real deployment.

## 8. Kubernetes / Minikube

First build the image:

```bash
docker build -t heart-disease-mlops:latest .
```

For Minikube:

```bash
minikube start
minikube image load heart-disease-mlops:latest
kubectl apply -f k8s/
kubectl get pods -n heart-mlops
kubectl get svc -n heart-mlops
```

The API service is a `LoadBalancer`.

For Minikube, use:

```bash
minikube tunnel
```

Then:

```bash
kubectl get svc -n heart-mlops
```

Use the external IP shown for the API.

Prometheus is exposed internally; Grafana is exposed with a LoadBalancer service. If using Minikube, `minikube service grafana -n heart-mlops` can open Grafana.

## 9. CI/CD

The GitHub Actions workflow runs:

1. dependency installation;
2. Ruff linting;
3. Pytest;
4. data download;
5. model training;
6. artifact upload.

Push this repository to GitHub and enable Actions.

## 10. Evidence/screenshots to capture

Place your own screenshots in `screenshots/`:

- EDA histograms
- correlation heatmap
- class balance
- MLflow experiment runs
- MLflow metrics/artifacts
- passing Pytest
- successful GitHub Actions run
- Docker build
- `/health`
- `/predict`
- `/metrics`
- Kubernetes pods/services
- Grafana dashboard
- final API endpoint

## 11. Submission checklist

- [ ] GitHub repository
- [ ] code
- [ ] requirements.txt
- [ ] dataset download script
- [ ] cleaned dataset
- [ ] EDA scripts/plots
- [ ] training script
- [ ] inference/API code
- [ ] tests
- [ ] GitHub Actions workflow
- [ ] Dockerfile
- [ ] Kubernetes manifests
- [ ] screenshots
- [ ] 10-page report
- [ ] deployed API URL or local access instructions
- [ ] short pipeline video

