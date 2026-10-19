"""Load the saved pipeline and score records.

Usage:  python -m src.predict '{"age": 63, "sex": 1, ...}'
"""
import json
import sys

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_FILE


def load_model(path=MODEL_FILE):
    return joblib.load(path)


def predict(model, records: list[dict]) -> list[dict]:
    X = pd.DataFrame(records)[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        label = int(p >= 0.5)
        out.append({
            "prediction": label,
            "label": "disease" if label else "no_disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if label else 1 - p), 4),
        })
    return out


if __name__ == "__main__":
    record = json.loads(sys.argv[1])
    print(json.dumps(predict(load_model(), [record])[0], indent=2))
