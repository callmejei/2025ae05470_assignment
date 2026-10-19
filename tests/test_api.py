import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import api.main as api_main
from src.config import FEATURES, TARGET
from src.data import clean
from src.features import build_pipeline


@pytest.fixture
def client(raw_df, tmp_path, monkeypatch):
    df = clean(raw_df)
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(df[FEATURES], df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(api_main, "MODEL_PATH", path)
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert "x-request-id" in r.headers


def test_predict_rejects_missing_field(client, sample_patient):
    bad = {k: v for k, v in sample_patient.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range(client, sample_patient):
    assert client.post("/predict",
                       json={**sample_patient, "age": 500}).status_code == 422


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
