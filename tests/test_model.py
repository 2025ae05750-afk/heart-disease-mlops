import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.train import build_preprocessor
from src.config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def test_preprocessor_fits_and_transforms():
    rows = []
    for i in range(10):
        row = {}
        for c in NUMERIC_FEATURES:
            row[c] = float(i + 1)
        for c in CATEGORICAL_FEATURES:
            row[c] = int(i % 2)
        rows.append(row)
    X = pd.DataFrame(rows, columns=FEATURES)
    X.loc[0, "chol"] = np.nan

    transformer = build_preprocessor()
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == 10
    assert transformed.shape[1] > len(FEATURES)
