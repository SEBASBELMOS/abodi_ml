import json
import joblib
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from fastapi import FastAPI, HTTPException
from api.schemas import ProcesoInput, BatchInput, RiskOutput, RiskFactor, BatchOutput, HealthOutput, ModelInfo
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(
    title="Abodi ML - Predictor de Vencimiento de Términos",
    description="API de inferencia para el modelo de riesgo de vencimiento de términos judiciales",
    version="1.0.0",
)

PROCESSED_PATH = Path("data/processed")

model = None
feature_cols = None
scaler = None
num_idx = None
metadata = None
plazo_map = None
despacho_order = None

prediction_counter = Counter("predictions_total", "Total de predicciones realizadas")
risk_counter = Counter("risk_detected_total", "Total de predicciones con riesgo detectado")
latency_histogram = Histogram("prediction_latency_seconds", "Latencia de predicción en segundos",
                               buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])

def load_model():
    global model, feature_cols, scaler, num_idx, metadata, plazo_map, despacho_order
    try:
        model_path = PROCESSED_PATH / "best_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            model_uri = "models:/Abodi_Risk_Classifier/Production"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Modelo cargado desde MLflow: {model_uri}")

        feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")
        scaler = joblib.load(PROCESSED_PATH / "scaler.pkl")
        num_idx = joblib.load(PROCESSED_PATH / "num_idx.pkl")
        plazo_map = joblib.load(PROCESSED_PATH / "plazo_map.pkl")
        despacho_order = joblib.load(PROCESSED_PATH / "despacho_order.pkl")

        meta_path = PROCESSED_PATH / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        print("Modelo y artefactos cargados exitosamente")
        return True
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return False

@app.on_event("startup")
async def startup():
    load_model()

PLAN_LIMITS = {
    "Freemium": 5,
    "Pro": 50,
    "Premium": 200,
    "Firma": 1_000_000,
}

plt_map = {
    "Auto admisorio demanda": 14,
    "Traslado excepciones": 14,
    "Traslado recurso": 7,
    "Sentencia primera instancia": 14,
    "Notificación por estado": 4,
    "Auto de pruebas": 14,
    "Fijación audiencia": 7,
    "Providencia interlocutoria": 0,
    "Constancia secretarial": 0,
    "Oficio comisorio": 0,
}

DESPACHO_ORDER = [
    "Juzgado Civil Municipal", "Juzgado Penal Municipal", "Juzgado de Familia",
    "Juzgado Civil del Circuito", "Juzgado Laboral Municipal",
    "Juzgado Laboral del Circuito", "Juzgado Administrativo", "Tribunal Administrativo",
]

def preprocess_input(data: ProcesoInput) -> np.ndarray:
    if data.despacho in DESPACHO_ORDER:
        despacho_enc = DESPACHO_ORDER.index(data.despacho)
    else:
        despacho_enc = len(DESPACHO_ORDER)

    plazo = plt_map.get(data.tipo_ultima_actuacion, 0)
    if plazo > 0:
        pct_plazo = min(data.dias_sin_actividad / plazo, 5.0)
    else:
        pct_plazo = 0.0

    row = {
        "dias_sin_actividad": data.dias_sin_actividad,
        "num_partes": data.num_partes,
        "total_actuaciones": data.total_actuaciones,
        "frecuencia_actualizaciones": data.frecuencia_actualizaciones,
        "tiene_termino_legal": data.tiene_termino_legal,
        "despacho_encoded": despacho_enc,
        "pct_plazo_consumido": pct_plazo,
    }

    one_hot_prefixes = {
        "tipo_proceso": data.tipo_proceso,
        "ciudad": data.ciudad,
        "plan_suscripcion": data.plan_suscripcion,
        "tipo_ultima_actuacion": data.tipo_ultima_actuacion,
    }

    for col in feature_cols:
        if col not in row:
            found = False
            for prefix, value in one_hot_prefixes.items():
                expected = f"{prefix}_{value}"
                if col == expected:
                    row[col] = 1
                    found = True
                    break
            if not found:
                row[col] = 0

    df = pd.DataFrame([row])[feature_cols]
    X = df.values.astype(np.float64)
    X[:, num_idx] = scaler.transform(X[:, num_idx])
    return X

@app.post("/predict/risk", response_model=RiskOutput)
async def predict_risk(data: ProcesoInput):
    if model is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Modelo no disponible")

    import time
    start = time.time()

    try:
        X = preprocess_input(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando entrada: {e}")

    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)

    prediction_counter.inc()
    if pred == 1:
        risk_counter.inc()

    latency = time.time() - start
    latency_histogram.observe(latency)

    if proba >= 0.7:
        nivel = "Alto"
    elif proba >= 0.5:
        nivel = "Medio"
    else:
        nivel = "Bajo"

    factores = []

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values[0]

        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)

        indices = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
        for i in indices:
            direction = "aumenta_riesgo" if shap_vals[0, i] > 0 else "disminuye_riesgo"
            factores.append(RiskFactor(
                feature=feature_cols[i],
                impact=float(abs(shap_vals[0, i])),
                direction=direction,
            ))

        shap_risk_factor = RiskFactor(feature="SHAP_explicabilidad", impact=1.0, direction="explicacion_shap")
        if not factores:
            factores.append(shap_risk_factor)
    except Exception:
        pass

    return RiskOutput(
        riesgo=pred,
        probabilidad=round(float(proba), 4),
        nivel=nivel,
        factores_riesgo=factores,
    )

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(data: BatchInput):
    if model is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Modelo no disponible")

    resultados = []
    for proceso in data.procesos:
        try:
            X = preprocess_input(proceso)
            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= 0.5)

            prediction_counter.inc()
            if pred == 1:
                risk_counter.inc()

            if proba >= 0.7:
                nivel = "Alto"
            elif proba >= 0.5:
                nivel = "Medio"
            else:
                nivel = "Bajo"

            resultados.append(RiskOutput(
                riesgo=pred,
                probabilidad=round(float(proba), 4),
                nivel=nivel,
                factores_riesgo=[],
            ))
        except Exception as e:
            resultados.append(RiskOutput(
                riesgo=-1,
                probabilidad=0.0,
                nivel="Error",
                factores_riesgo=[],
            ))

    return BatchOutput(
        resultados=resultados,
        total=len(resultados),
        en_riesgo=sum(1 for r in resultados if r.riesgo == 1),
    )

@app.get("/health", response_model=HealthOutput)
async def health():
    model_ok = model is not None
    model_version = None
    model_f1 = None
    if metadata:
        model_version = f"v{metadata.get('version', '?')}"
        model_f1 = metadata.get("metrics", {}).get("f1_score")

    return HealthOutput(
        status="ok" if model_ok else "degradado",
        model_version=model_version,
        model_f1=model_f1,
        total_predictions=int(prediction_counter._value.get()),
    )

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    if metadata:
        return ModelInfo(
            model_name=metadata.get("model_name", "Abodi_Risk_Classifier"),
            version=f"v{metadata.get('version', '?')}",
            metrics=metadata.get("metrics", {}),
            features=feature_cols if feature_cols else [],
            fecha_entrenamiento=metadata.get("run_name", "desconocida"),
        )
    return ModelInfo(
        model_name="Abodi_Risk_Classifier",
        metrics={},
        features=feature_cols if feature_cols else [],
    )

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
