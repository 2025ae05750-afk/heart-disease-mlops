from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/heart_disease.csv"
PROCESSED_PATH = ROOT / "data/processed/heart_disease_clean.csv"
ARTIFACT_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACT_DIR / "plots"
METRICS_DIR = ARTIFACT_DIR / "metrics"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MLFLOW_DIR = ROOT / "mlruns"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"
