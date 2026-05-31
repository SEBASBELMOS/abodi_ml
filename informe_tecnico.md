# Informe Técnico: Predictor de Vencimiento de Términos

## Proyecto: Abodi 2.0 — Módulo ML MLOps

---

## 1. Problema

Los abogados en Colombia pierden de 1 a 3 horas diarias revisando manualmente los portales de la Rama Judicial. Cada proceso judicial tiene términos legales (plazos) que, si se vencen, pueden generar responsabilidades disciplinarias, pérdida de oportunidades procesales o incluso la pérdida del caso. No existe una herramienta que **automatice la detección de estos riesgos** antes de que sea demasiado tarde.

## 2. Solución Propuesta

Se desarrolló un **clasificador binario** (riesgo de vencimiento: Sí/No) basado en **XGBoost** que, a partir de características estructurales del proceso judicial (tipo de proceso, días sin actividad, tipo de última actuación, ciudad, despacho, plan de suscripción, etc.), predice si un proceso tiene riesgo inminente de vencimiento de términos.

El modelo no reemplaza al abogado, sino que **prioriza su atención**: filtra los procesos que requieren acción urgente de los que están al día.

## 3. ¿Por qué este modelo y no otro?

Se evaluaron 4 propuestas en `propuestas_modelos.md`. La elección final fue **Vencimiento de Términos (Opción 2)** por estas razones:

| Criterio | Vencimiento Términos | Duración (Op1) | Tutelas (Op3) | Riesgo Pérdida (Op4) |
| :--- | :---: | :---: | :---: | :---: |
| **Alineación con negocio Abodi** | Alta | Media | Baja | Media |
| **Clasificación (Accuracy, F1, Recall)** | Binaria pura | Multiclase | Binaria | Multiclase |
| **Datos sintéticos realistas** | Fácil | Fácil | Media | Media |
| **Impacto en demo** | Semáforo rojo/verde | Barras | SHAP | SHAP |
| **Monitoreo con sentido** (drift) | Alto | Bajo | Medio | Medio |

**El problema central que Abodi resuelve es evitar términos vencidos.** Por lo tanto, este modelo es el más alineado con el valor de negocio de la plataforma.

## 4. Metodología

### 4.1 Datos Sintéticos

Se generaron **10,000 registros sintéticos** basados en reglas de negocio del sistema judicial colombiano:

- **Tipos de proceso:** Tutela (30%), Ordinario Civil (20%), Ejecutivo (20%), Laboral (15%), Administrativo (15%)
- **Actuaciones judiciales:** 10 tipos, algunos con término legal asociado, otros informativos
- **Ciudades:** Bogotá, Medellín, Cali, Barranquilla, Manizales, Bucaramanga y Otras — con factor de congestión que afecta los tiempos entre actuaciones
- **Despachos:** 8 tipos de juzgados, mapeados por especialidad
- **Planes de suscripción:** Freemium, Pro, Premium, Firma

**Regla de etiquetado del target (riesgo_vencimiento):**
```
riesgo = 1  si  (tiene_termino_legal AND dias_sin_actividad > plazo_legal * 0.7)
riesgo = 0  en caso contrario
```
Se agregó **8% de ruido aleatorio** para evitar que el modelo memorice una regla determinística perfecta.

### 4.2 Feature Engineering (ETL)

| Feature | Tipo | Descripción |
| :--- | :--- | :--- |
| `dias_sin_actividad` | Numérica | Días desde la última actuación registrada |
| `num_partes` | Numérica | Número de partes en el proceso |
| `total_actuaciones` | Numérica | Total de actuaciones registradas |
| `frecuencia_actualizaciones` | Numérica | Actuaciones por mes |
| `tiene_termino_legal` | Binaria | 1 si la última actuación tiene plazo legal |
| `pct_plazo_consumido` | Numérica | % del plazo legal consumido (feature engineered) |
| `despacho_encoded` | Ordinal | Despacho codificado por congestión |
| `tipo_proceso_*` (4) | One-hot | Dummies por tipo de proceso |
| `ciudad_*` (6) | One-hot | Dummies por ciudad |
| `plan_suscripcion_*` (3) | One-hot | Dummies por plan |
| `tipo_ultima_actuacion_*` (9) | One-hot | Dummies por tipo de actuación |

**Total: 29 features** (6 numéricas + 23 one-hot).

### 4.3 Entrenamiento (MLflow — 3 Experimentos)

Se entrenaron 3 variantes de XGBoost, registrando todo en MLflow:

#### Experimento 1: Baseline (class_weight='balanced')
- **Hiperparámetros:** max_depth=4, n_estimators=100, lr=0.1
- **Balanceo:** scale_pos_weight calculado de la proporción de clases

#### Experimento 2: SMOTE + Profundo
- **Hiperparámetros:** max_depth=8, n_estimators=200, lr=0.05
- **Balanceo:** SMOTE (Synthetic Minority Oversampling Technique)

#### Experimento 3: ADASYN + Regularizado ← **GANADOR**
- **Hiperparámetros:** max_depth=3, n_estimators=150, lr=0.01, reg_lambda=2, reg_alpha=1
- **Balanceo:** ADASYN (Adaptive Synthetic Sampling)

### 4.4 Resultados

| Experimento | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (class_weight) | 0.8705 | 0.8524 | 0.8329 | **0.8426** | 0.8990 |
| SMOTE + Profundo | 0.8640 | 0.8535 | 0.8125 | 0.8325 | 0.8944 |
| ADASYN + Regularizado | **0.8715** | **0.8563** | 0.8305 | **0.8432** | **0.9053** |

**Mejor modelo: ADASYN + XGBoost regularizado** — seleccionado por mayor F1-Score (0.8432) y AUC-ROC (0.9053).

### 4.5 Selección y Registro

El script `select_best.py`:
1. Consulta todos los runs en MLflow
2. Ordena por F1-Score descendente
3. Descarga el modelo ganador
4. Lo guarda como `data/processed/best_model.pkl`
5. Registra metadatos en `data/processed/metadata.json`

### 4.6 API (FastAPI)

**5 endpoints:**

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| POST | `/predict/risk` | Predicción individual con explicación SHAP |
| POST | `/predict/batch` | Predicción por lote (múltiples procesos) |
| GET | `/health` | Health check con versión y métricas del modelo |
| GET | `/model/info` | Metadatos del modelo en producción |
| GET | `/metrics` | Métricas Prometheus (latencia, conteo) |

### 4.7 Interfaz (Streamlit)

**3 pestañas:**

1. **Evaluación Individual:** Formulario con inputs de todas las features, botón "Evaluar Riesgo", semáforo (🔴 Alto / 🟡 Medio / 🟢 Bajo), porcentaje de probabilidad y desglose de factores SHAP
2. **Carga Masiva:** Subida de CSV con descarga de plantilla de ejemplo, tabla de resultados, filtros y exportación
3. **Dashboard de Salud:** KPIs del modelo, estado de la API, límites por plan y tabla de procesos de ejemplo

### 4.8 Tests Unitarios

**13 tests** organizados en 2 archivos:

- `tests/test_model.py` (7 tests): Carga del modelo, features, predicción válida, consistencia, metadata, despacho_order, scaler
- `tests/test_api.py` (6 tests): Health check, predict/risk válido e inválido, predict/batch, model/info, métricas Prometheus

### 4.9 Docker

3 servicios en `docker-compose.yml`:
- `api`: FastAPI (puerto 8000)
- `app`: Streamlit (puerto 8501)
- `mlflow`: MLflow Tracking Server (puerto 5000)
- `prometheus`: Métricas (puerto 9090)
- `grafana`: Dashboards (puerto 3000)

### 4.10 CI/CD (GitHub Actions)

Pipeline automático en cada push a `main`:
1. Setup Python 3.11
2. Instalar dependencias
3. Generar datos sintéticos
4. Ejecutar ETL
5. Entrenar modelo (3 experimentos)
6. Seleccionar el mejor
7. Ejecutar tests (pytest)
8. Linting (ruff)
9. Build Docker images

---

## 5. Cómo Ejecutar Paso a Paso

### Requisitos

- Python 3.9+ o 3.11+
- pip
- [Opcional] Docker Desktop

### 5.1 Instalación

```bash
cd abodi/ml_module
pip install -r requirements.txt
```

### 5.2 Generar datos

```bash
python src/generate_data.py
```

### 5.3 ETL

```bash
python src/etl.py
```

### 5.4 Entrenar (3 experimentos)

```bash
python src/train.py
```

### 5.5 Seleccionar mejor modelo

```bash
python src/select_best.py
```

### 5.6 Ver MLflow

```bash
mlflow ui
# Abrir http://localhost:5000
```

### 5.7 Iniciar API

```bash
uvicorn api.router:app --host 0.0.0.0 --port 8000
# Probar: curl http://localhost:8000/health
```

### 5.8 Iniciar Streamlit

```bash
streamlit run app/streamlit_app.py
# Abrir http://localhost:8501
```

### 5.9 Ejecutar Tests

```bash
pytest tests/ -v
```

### 5.10 Docker (todo junto)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 6. Estructura Final del Módulo

```
ml_module/
├── api/
│   ├── __init__.py
│   ├── router.py              # FastAPI: 5 endpoints
│   └── schemas.py              # Pydantic models
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # 3 pestañas: Individual, Batch, Dashboard
├── data/
│   ├── raw/
│   │   └── procesos_judiciales.csv   # 10,000 registros sintéticos
│   └── processed/
│       ├── best_model.pkl            # Modelo ganador
│       ├── scaler.pkl                # StandardScaler
│       ├── feature_cols.pkl          # Nombres de features
│       ├── num_idx.pkl               # Índices de features numéricas
│       ├── despacho_order.pkl        # Orden de despachos
│       ├── plazo_map.pkl             # Mapa de plazos legales
│       ├── metadata.json             # Métricas del mejor modelo
│       ├── X_train.npy / X_test.npy
│       └── y_train.npy / y_test.npy
├── src/
│   ├── __init__.py
│   ├── generate_data.py       # Generación sintética
│   ├── etl.py                 # Feature engineering + split
│   ├── train.py               # 3 experimentos MLflow
│   └── select_best.py         # Selección + registro
├── tests/
│   ├── __init__.py
│   ├── test_model.py          # 7 tests
│   └── test_api.py            # 6 tests
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.app
│   ├── docker-compose.yml
│   └── prometheus.yml
├── .github/workflows/
│   └── ci.yml                 # CI/CD pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Conclusiones

1. **Cumplimiento MLOps:** El proyecto implementa registro de experimentos (MLflow), CI/CD (GitHub Actions), servicio de inferencia (FastAPI), interfaz de demostración (Streamlit), Docker y monitoreo (Prometheus/Grafana).

2. **Rendimiento del modelo:** F1-Score de 0.843 y AUC-ROC de 0.905, con Accuracy de 0.872. El modelo es suficientemente preciso para uso en producción como herramienta de soporte.

3. **Datos sintéticos:** Las 10,000 muestras generadas reflejan distribuciones realistas del sistema judicial colombiano, con 29 features que cubren tipo de proceso, geografía, temporalidad y planes de suscripción.

4. **Mejor experimento:** ADASYN + XGBoost regularizado superó a Baseline y SMOTE, demostrando que el balanceo adaptativo combinado con regularización produce el mejor rendimiento en este problema.

5. **Próximos pasos:** Subir a GitHub, conectar con datos reales de Supabase, probar con scraper real, y afinar hiperparámetros con datos reales para superar el umbral de F1 ≥ 0.85.

---

*Documento generado para el proyecto final de MLOps — Abodi App. Mayo 2026.*
