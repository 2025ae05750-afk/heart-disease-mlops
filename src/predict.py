import argparse
import json

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH


def predict_one(payload: dict):
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probability_absence": float(probabilities[0]),
        "probability_presence": float(probabilities[1]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON object containing 13 feature values")
    args = parser.parse_args()
    payload = json.loads(args.input)
    print(json.dumps(predict_one(payload), indent=2))

if __name__ == "__main__":
    main()
