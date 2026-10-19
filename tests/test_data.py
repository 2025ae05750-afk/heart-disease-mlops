import pandas as pd
from src.data import clean_data
from src.config import FEATURES, TARGET

def test_clean_data_converts_target_and_preserves_features():
    data = {c: [1, 2, 3] for c in FEATURES}
    data[TARGET] = [0, 1, 4]
    df = clean_data(pd.DataFrame(data))
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df[TARGET].unique()) == {0, 1}

def test_clean_data_drops_missing_target():
    data = {c: [1, 2] for c in FEATURES}
    data[TARGET] = [0, None]
    df = clean_data(pd.DataFrame(data))
    assert len(df) == 1
