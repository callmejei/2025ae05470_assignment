import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import train as train_mod
from src.config import FEATURES, TARGET
from src.data import clean
from src.features import build_pipeline, build_preprocessor
from src.predict import predict


def test_preprocessor_output_is_scaled_and_encoded(raw_df):
    df = clean(raw_df)
    X = build_preprocessor().fit_transform(df[FEATURES])
    assert X.shape[0] == len(df)
    assert X.shape[1] > len(FEATURES)  # one-hot expands categoricals
    assert not np.isnan(np.asarray(X.todense() if hasattr(X, "todense")
                                   else X)).any()


def test_pipeline_handles_unseen_category(raw_df, sample_patient):
    df = clean(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(df[FEATURES], df[TARGET])
    odd = {**sample_patient, "thal": 99}
    out = predict(model, [odd])[0]
    assert 0 <= out["probability_disease"] <= 1


def test_predict_output_contract(raw_df, sample_patient):
    df = clean(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(df[FEATURES], df[TARGET])
    out = predict(model, [sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert out["label"] in ("disease", "no_disease")
    assert 0.5 <= out["confidence"] <= 1


def test_train_end_to_end(raw_df, tmp_path, monkeypatch):
    data = tmp_path / "clean.csv"
    clean(raw_df).to_csv(data, index=False)
    monkeypatch.setattr(train_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "MODEL_FILE", tmp_path / "model.joblib")
    monkeypatch.setattr(train_mod, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(train_mod, "ROOT", tmp_path)
    monkeypatch.setattr(train_mod, "CANDIDATES", {
        "logistic_regression": (LogisticRegression(max_iter=500),
                                {"model__C": [1.0]}),
    })

    summary = train_mod.train(data, tracking_uri=(tmp_path / "mlruns").as_uri())

    assert summary["best_model"] == "logistic_regression"
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert 0 <= saved["models"]["logistic_regression"]["cv_roc_auc"] <= 1
    model = joblib.load(tmp_path / "model.joblib")
    assert model.predict(clean(raw_df)[FEATURES]).shape == (len(raw_df),)
