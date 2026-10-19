import json
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def main():
    print("Downloading UCI Heart Disease dataset (dataset ID 45)...")
    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Normalize feature names while preserving the assignment's 13 clinical inputs.
    if len(X.columns) != 13:
        raise ValueError(f"Expected 13 features, found {len(X.columns)}")

    X.columns = COLUMN_NAMES

    target_col = y.columns[0]
    target = pd.to_numeric(y[target_col], errors="coerce")

    # UCI's goal is 0 = no disease and 1-4 = disease presence.
    target = target.map(lambda v: 1 if v in [1, 2, 3, 4] else 0 if v == 0 else pd.NA)

    df = X.copy()
    df["target"] = target

    # Convert common UCI missing-value markers to NaN.
    df = df.replace(["?", "NA", "N/A", ""], pd.NA)

    # Ensure numeric representation where possible.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(RAW_DIR / "heart_disease.csv", index=False)

    metadata = {
        "source": "UCI Machine Learning Repository",
        "dataset_id": 45,
        "dataset": "Heart Disease",
        "subset_used": "Cleveland / processed UCI features",
        "features": COLUMN_NAMES,
        "target": "target",
        "target_definition": "0 = absence, 1 = presence",
        "citation": "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository.",
        "doi": "10.24432/C52P4X",
    }
    (RAW_DIR / "heart_disease_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(df)} rows to {RAW_DIR / 'heart_disease.csv'}")
    print(df.head())

if __name__ == "__main__":
    main()
