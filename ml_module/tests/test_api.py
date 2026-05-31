import pytest
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")
num_idx_orig = joblib.load(PROCESSED_PATH / "num_idx.pkl")

scaler_orig = joblib.load(PROCESSED_PATH / "scaler.pkl")

mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
try:
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    y_train = np.load(PROCESSED_PATH / "y_train.npy")
    mock_model.fit(X_train, y_train)
except Exception:
    mock_model.fit(np.random.randn(100, len(feature_cols)), np.random.randint(0, 2, 100))
joblib.dump(mock_model, PROCESSED_PATH / "best_model.pkl")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.router import app
from fastapi.testclient import TestClient

client = TestClient(app)

def make_payload(overrides=None):
    payload = {
        "tipo_proceso": "Tutela",
        "tipo_ultima_actuacion": "Notificación por estado",
        "ciudad": "Bogotá",
        "despacho": "Juzgado Civil Municipal",
        "dias_sin_actividad": 10,
        "num_partes": 2,
        "total_actuaciones": 5,
        "frecuencia_actualizaciones": 1.5,
        "tiene_termino_legal": 1,
        "plan_suscripcion": "Pro",
    }
    if overrides:
        payload.update(overrides)
    return payload


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_risk_valid():
    payload = make_payload()
    response = client.post("/predict/risk", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    assert "riesgo" in data
    assert "probabilidad" in data
    assert "nivel" in data
    assert data["riesgo"] in [0, 1]


def test_predict_risk_invalid():
    payload = {"tipo_proceso": ""}
    response = client.post("/predict/risk", json=payload)
    assert response.status_code == 422


def test_predict_batch():
    payload = {
        "procesos": [
            make_payload(overrides={"dias_sin_actividad": 5}),
            make_payload(overrides={"tipo_proceso": "Ordinario Civil", "dias_sin_actividad": 30}),
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    assert "resultados" in data
    assert data["total"] == 2
    assert "en_riesgo" in data


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
