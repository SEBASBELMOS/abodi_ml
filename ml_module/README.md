# Abodi ML — Predictor de Vencimiento de Términos

Módulo de Machine Learning para la plataforma **Abodi 2.0** que predice el riesgo de vencimiento de términos en procesos judiciales colombianos.

## Problema

Los abogados en Colombia pierden de 1 a 3 horas diarias revisando manualmente los portales de la Rama Judicial, con alto riesgo de omitir actuaciones críticas y vencimiento de términos. Este modelo automatiza la detección de procesos en riesgo.

## Solución

Clasificador binario (XGBoost) que, dado un proceso judicial, predice si existe riesgo inminente de vencimiento de términos. El modelo se entrena con datos sintéticos basados en reglas de negocio del sistema judicial colombiano y se sirve mediante una API FastAPI integrable al backend existente de Abodi.

## Arquitectura

```
Usuario → Streamlit App → FastAPI API → Modelo XGBoost → Predicción + SHAP
                ↓                                            ↓
         Dashboard Salud                              MLflow Tracking
                ↓                                            ↓
          Grafana/Prometheus                        Model Registry
```

## Requisitos

- Python 3.11+
- pip

## Instalación

```bash
cd ml_module
pip install -r requirements.txt
```

## Uso Paso a Paso

### 1. Generar datos sintéticos

```bash
python src/generate_data.py
```

Genera 10,000 registros sintéticos en `data/raw/procesos_judiciales.csv`.

### 2. ETL y Feature Engineering

```bash
python src/etl.py
```

Transforma datos, aplica one-hot encoding y escalado. Guarda en `data/processed/`.

### 3. Entrenar modelo (3 experimentos con MLflow)

```bash
python src/train.py
```

Ejecuta 3 experimentos XGBoost con diferentes hiperparámetros y técnicas de balanceo. Todos los resultados se registran en MLflow.

### 4. Seleccionar el mejor modelo

```bash
python src/select_best.py
```

Compara los 3 experimentos por F1-Score, selecciona el mejor y lo registra en MLflow Model Registry como "Production".

### 5. Ver experimentos en MLflow

```bash
mlflow ui
```

Abrir `http://localhost:5000`.

### 6. Iniciar API

```bash
uvicorn api.router:app --host 0.0.0.0 --port 8000
```

Endpoints disponibles:
- `POST /predict/risk` — predicción individual
- `POST /predict/batch` — predicción por lote
- `GET /health` — health check
- `GET /model/info` — metadatos del modelo
- `GET /metrics` — métricas Prometheus

### 7. Iniciar interfaz Streamlit

```bash
streamlit run app/streamlit_app.py
```

Abrir `http://localhost:8501`.

## Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Servicios:
- **API**: `http://localhost:8000`
- **Streamlit**: `http://localhost:8501`
- **MLflow**: `http://localhost:5000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

## Tests

```bash
pytest tests/ -v
```

## CI/CD

GitHub Actions corre automáticamente en cada push a main:
1. Genera datos sintéticos
2. Ejecuta ETL
3. Entrena modelo (3 experimentos)
4. Selecciona el mejor
5. Corre tests unitarios
6. Linting con ruff
7. Build de imágenes Docker

## KPIs del Modelo

| Métrica | Objetivo |
| :--- | :--- |
| Accuracy | ≥ 0.85 |
| Precision | ≥ 0.80 |
| Recall | ≥ 0.80 |
| F1-Score | ≥ 0.85 |
| AUC-ROC | ≥ 0.90 |

## Integración con Backend Abodi

Los endpoints de la API (`api/router.py`) están diseñados para importarse directamente desde el backend existente de Abodi:

```python
# backend/main.py
from ml_module.api.router import router as ml_router
app.include_router(ml_router, prefix="/ml")
```

## Estructura del Módulo

```
ml_module/
├── api/               # FastAPI endpoints
│   ├── router.py      # Endpoints de inferencia
│   └── schemas.py     # Modelos Pydantic
├── app/               # Streamlit (demo)
│   └── streamlit_app.py
├── data/              # Datos sintéticos
│   ├── raw/           # Datos generados
│   └── processed/     # Datos transformados + modelo
├── src/               # Scripts de entrenamiento
│   ├── generate_data.py
│   ├── etl.py
│   ├── train.py
│   └── select_best.py
├── tests/             # Tests unitarios
│   ├── test_model.py
│   └── test_api.py
├── docker/            # Contenedores
│   ├── Dockerfile.api
│   ├── Dockerfile.app
│   └── docker-compose.yml
├── .github/workflows/ # CI/CD
├── requirements.txt
└── README.md
```
