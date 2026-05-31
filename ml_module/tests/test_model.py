import pytest
import numpy as np
import joblib
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def test_modelo_cargado():
    model_path = PROCESSED_PATH / "best_model.pkl"
    assert model_path.exists(), "El modelo no existe. Ejecuta select_best.py primero"
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_features_cargadas():
    features_path = PROCESSED_PATH / "feature_cols.pkl"
    assert features_path.exists()
    features = joblib.load(features_path)
    assert isinstance(features, list)
    assert len(features) > 0


def test_prediccion_valida():
    model = joblib.load(PROCESSED_PATH / "best_model.pkl")
    feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")
    scaler = joblib.load(PROCESSED_PATH / "scaler.pkl")
    num_idx = joblib.load(PROCESSED_PATH / "num_idx.pkl")

    row = {col: 0.0 for col in feature_cols}
    row["dias_sin_actividad"] = 15.0
    row["num_partes"] = 2.0
    row["total_actuaciones"] = 5.0
    row["frecuencia_actualizaciones"] = 1.5
    row["tiene_termino_legal"] = 1.0
    row["despacho_encoded"] = 0.0
    row["pct_plazo_consumido"] = 0.5

    import pandas as pd
    df = pd.DataFrame([row])[feature_cols]
    X = df.values.astype(np.float64)

    X[:, num_idx] = scaler.transform(X[:, num_idx])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    assert pred in [0, 1]
    assert 0 <= proba[0] <= 1
    assert 0 <= proba[1] <= 1


def test_prediccion_consistente():
    model = joblib.load(PROCESSED_PATH / "best_model.pkl")
    feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")
    scaler = joblib.load(PROCESSED_PATH / "scaler.pkl")
    num_idx = joblib.load(PROCESSED_PATH / "num_idx.pkl")

    row = {col: 0.0 for col in feature_cols}
    row["dias_sin_actividad"] = 30.0
    row["num_partes"] = 3.0
    row["total_actuaciones"] = 10.0
    row["frecuencia_actualizaciones"] = 2.0
    row["tiene_termino_legal"] = 1.0
    row["despacho_encoded"] = 1.0
    row["pct_plazo_consumido"] = 1.5

    import pandas as pd
    df = pd.DataFrame([row])[feature_cols]
    X = df.values.astype(np.float64)
    X[:, num_idx] = scaler.transform(X[:, num_idx])

    pred1 = model.predict(X)[0]
    proba1 = model.predict_proba(X)[0, 1]
    pred2 = model.predict(X)[0]
    proba2 = model.predict_proba(X)[0, 1]

    assert pred1 == pred2
    assert proba1 == proba2


def test_metadata():
    meta_path = PROCESSED_PATH / "metadata.json"
    assert meta_path.exists()
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert "metrics" in meta
    assert "f1_score" in meta["metrics"]


def test_despacho_order():
    order_path = PROCESSED_PATH / "despacho_order.pkl"
    assert order_path.exists()
    order = joblib.load(order_path)
    assert isinstance(order, list)
    assert len(order) > 0


def test_scaler_cargado():
    scaler_path = PROCESSED_PATH / "scaler.pkl"
    assert scaler_path.exists()
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")
