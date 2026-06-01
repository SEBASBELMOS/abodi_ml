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
# Desde la raíz del repo (requirements.txt está en la raíz)
pip install -r requirements.txt
cd ml_module
```

> Los comandos de la sección siguiente se ejecutan desde `ml_module/`.

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

## Resultados

**Modelo ganador:** ADASYN + XGBoost regularizado (`max_depth=3`, `n_estimators=150`, `lr=0.01`, `reg_lambda=2`, `reg_alpha=1`), seleccionado por mayor F1-Score entre los 3 experimentos.

| Métrica | Objetivo | Obtenido | Estado |
| :--- | :---: | :---: | :---: |
| Accuracy | ≥ 0.85 | 0.9100 | ✅ |
| Precision | ≥ 0.80 | 0.9158 | ✅ |
| Recall | ≥ 0.80 | 0.8630 | ✅ |
| F1-Score | ≥ 0.85 | 0.8886 | ✅ |
| AUC-ROC | ≥ 0.90 | 0.9126 | ✅ |

El modelo supera todos los objetivos. Las métricas se calculan sobre el conjunto de test (split 80/20 estratificado) y quedan registradas en MLflow.

## Integración con Backend Abodi

La API (`api/router.py`) expone una aplicación FastAPI (`app`) lista para integrarse con el backend existente de Abodi de dos formas:

```python
# Opción A (recomendada): correr el módulo como microservicio independiente
#   y consumirlo por HTTP desde el backend (POST http://ml:8000/predict/risk)

# Opción B: montar la app del modelo como sub-aplicación del backend
# backend/main.py
from ml_module.api.router import app as ml_app
main_app.mount("/ml", ml_app)
```

## Estructura del Módulo

```
abodi_ml/                  # raíz del repo
├── README.md
├── requirements.txt       # dependencias del módulo ML
├── docs/                  # documentación del proyecto
└── ml_module/
    ├── api/               # FastAPI
    │   ├── router.py      # App FastAPI con los endpoints de inferencia
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
    │   ├── Dockerfile.mlflow
    │   ├── prometheus.yml
    │   └── docker-compose.yml
    ├── mlruns/            # Tracking local de MLflow (generado)
    └── .github/workflows/ # CI/CD (ci.yml)
```
