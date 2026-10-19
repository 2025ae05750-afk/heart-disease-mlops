from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURES, PROCESSED_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first."
        )
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Treat UCI placeholders as missing and coerce numeric columns.
    df = df.replace(["?", "NA", "N/A", ""], pd.NA)
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where target is unavailable; feature missing values are
    # intentionally imputed inside the sklearn pipeline to avoid leakage.
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        # Supports raw UCI target coding 0..4 if a raw file is supplied.
        df[TARGET] = df[TARGET].apply(lambda x: 1 if x > 0 else 0)

    return df[FEATURES + [TARGET]]

def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

def save_clean_data(df: pd.DataFrame):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH
