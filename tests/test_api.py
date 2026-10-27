import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from api.schemas import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

VALID = {
    "tenure_months": 12,
    "avg_monthly_spend_ngn": 5000,
    "recharge_frequency_per_month": 4,
    "payment_consistency_score": 0.2,
    "late_payment_count": 6,
    "operator": "MTN",
    "account_type": "prepaid",
}


def _toy_model() -> Pipeline:
    """Petit modèle synthétique : les tests ne dépendent pas des données DVC."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 120, n),
            "avg_monthly_spend_ngn": rng.uniform(500, 50_000, n),
            "recharge_frequency_per_month": rng.integers(1, 30, n),
            "payment_consistency_score": rng.uniform(0, 1, n),
            "late_payment_count": rng.integers(0, 10, n),
            "operator": rng.choice(["MTN", "Airtel", "Glo", "9mobile"], n),
            "account_type": rng.choice(["prepaid", "postpaid"], n),
        }
    )
    y = (df["late_payment_count"] + rng.normal(0, 2, n) > 5).astype(int)
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(df[FEATURES], y)
    return pipe


@pytest.fixture()
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_toy_model(), model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))

    from api.main import app

    with TestClient(app) as c:
        c.log_path = tmp_path / "predictions.jsonl"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_predict_returns_probability_and_logs(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert isinstance(body["is_high_risk"], bool)

    lines = client.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == body["request_id"]
    assert "customer_id" not in record["features"]
    assert "phone_number" not in record["features"]


def test_rejects_out_of_range_value(client):
    r = client.post("/predict", json={**VALID, "payment_consistency_score": 1.7})
    assert r.status_code == 422


def test_rejects_unknown_operator(client):
    r = client.post("/predict", json={**VALID, "operator": "Orange"})
    assert r.status_code == 422


def test_rejects_identifier_field(client):
    r = client.post("/predict", json={**VALID, "customer_id": "CUST1234567"})
    assert r.status_code == 422


def test_features_match_training_config():
    from src import config

    assert FEATURES == config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES