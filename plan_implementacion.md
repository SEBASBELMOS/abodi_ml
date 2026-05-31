# Plan de Implementación: Predictor de Vencimiento de Términos

Basado en la **Opción 2 (Vencimiento de Términos)** de `propuestas_modelos.md` y alineado con el contexto real de Abodi 2.0 en `context.md`.

---

## ¿Qué necesitamos de la aplicación (Abodi App)?

### Conocimiento del dominio legal colombiano
- **Tipos de procesos judiciales** y sus plazos legales típicos (Tutela: 3-10 días, Ordinario Civil: 10-30 días, Ejecutivo: 5-15 días, Laboral: 10-20 días)
- **Tipos de actuaciones judiciales** con término asociado (Auto interlocutorio, Traslado, Notificación, Sentencia, Providencia)
- **Estructura del radicado de 23 dígitos** — ciudad (dígitos 1-5), despacho (6-9), año, especialidad

### Stack real de Abodi que impacta el modelo
| Componente | Cómo impacta al modelo ML |
| :--- | :--- |
| **Supabase (PostgreSQL, 28+ tablas)** | Allí se almacenan históricos de procesos, actuaciones y estados. El modelo consumiría datos desde aquí en producción |
| **Backend FastAPI existente** | Los endpoints del modelo se integran como nuevos routers dentro de este backend, no como servicio separado |
| **Vue 3 + Vite (Frontend SaaS)** | La predicción del modelo se mostraría dentro del SaaS real; Streamlit es solo para demo/prototipo |
| **Scrapers (Webshare + scripts)** | Deben ejecutarse 4x/día (8AM, 11AM, 2PM, 5PM) y alimentar la BD con datos actualizados. El modelo depende de estos datos frescos |
| **WebSockets + Resend (notificaciones)** | Canal para alertar al usuario cuando el modelo detecta riesgo de vencimiento |
| **Planes de suscripción** | El modelo debe respetar límites de procesos por plan (Freemium: 5, Pro: 50, Premium: 200, Firma: ilimitado) |

### Features que usará el modelo
| Feature | Fuente en Abodi |
| :--- | :--- |
| Días desde última actuación | Tabla `actuaciones` — resta entre fecha actual y última actuación |
| Tipo de proceso | Tabla `procesos` — columna `tipo_proceso` |
| Tipo de última actuación | Tabla `actuaciones` — columna `tipo_actuacion` |
| Despacho / Juzgado | Tabla `procesos` — columna `despacho_id` (relacionado con radicado) |
| Ciudad | Tabla `despachos` — join con ciudad/departamento |
| Tiene término legal asociado | Regla de negocio: ciertos tipos de actuación tienen plazo legal |
| Número de partes | Tabla `partes_proceso` — count por proceso_id |
| Cantidad de actuaciones previas | Tabla `actuaciones` — count por proceso_id |
| Plan de suscripción del usuario | Tabla `suscripciones` — afecta si el proceso está activo o no |
| Frecuencia de actualizaciones (act/mes) | Tabla `actuaciones` — avg rate de los últimos 90 días |

### Lo que NO necesitas (para este entregable)
- No necesitas acceso real a la BD de Supabase (todo es sintético)
- No necesitas datos de scraping reales
- No necesitas la app Vue 3 funcionando
- No necesitas WebSockets reales

> Todo el dataset será **sintético** pero modelado a partir de las estructuras de BD reales de Abodi.

---

## Estructura del Módulo ML (dentro del repositorio Abodi)

```
abodi/                          # ← Repositorio raíz de Abodi 2.0
├── backend/                    # FastAPI existente del SaaS
│   └── routers/                # Aquí se integrarán los endpoints del modelo
├── frontend/                   # Vue 3 + Vite (SaaS real)
├── scripts/                    # Scrapers existentes
├── supabase/                   # Migraciones y esquemas
├── docker/                     # Docker real de producción
│
└── ml_module/                  # ← Este entregable MLOps
    ├── api/
    │   ├── __init__.py
    │   ├── router.py           # Endpoints FastAPI (se copian a backend/routers/)
    │   └── schemas.py          # Modelos Pydantic
    ├── app/
    │   ├── __init__.py
    │   └── streamlit_app.py    # Prototipo demo (consume la API)
    ├── data/
    │   ├── raw/                # Datos sintéticos generados (procesos_judiciales.csv)
    │   └── processed/          # Datos limpios + mejor modelo (.pkl)
    ├── src/
    │   ├── __init__.py
    │   ├── generate_data.py    # Generación de datos sintéticos
    │   ├── etl.py              # Limpieza y transformación
    │   ├── train.py            # Entrenamiento con MLflow (3 experimentos)
    │   └── select_best.py      # Selección y registro en Model Registry
    ├── tests/
    │   ├── __init__.py
    │   ├── test_api.py         # Tests de endpoints
    │   └── test_model.py       # Tests de predicción
    ├── docker/
    │   ├── Dockerfile.api
    │   ├── Dockerfile.app
    │   └── docker-compose.yml  # api + app + mlflow + prometheus + grafana
    ├── requirements.txt
    └── README.md
```

---

## Checklist de Implementación Paso a Paso

### Fase 0: Setup del Proyecto
- [ ] Crear carpeta `ml_module/` dentro del repositorio Abodi
- [ ] Crear estructura interna (api/, app/, data/, src/, tests/, docker/)
- [ ] Inicializar tracking local de MLflow (carpeta `mlruns/`)
- [ ] Crear `requirements.txt` con dependencias (pandas, scikit-learn, xgboost, mlflow, fastapi, uvicorn, streamlit, shap, pytest, imbalanced-learn, prometheus-client)
- [ ] Crear `.gitignore` dentro de `ml_module/`
- [ ] Revisar esquemas de Supabase (tablas `procesos`, `actuaciones`, `partes_proceso`, `suscripciones`) para modelar datos sintéticos

### Fase 1: Generación de Datos Sintéticos
- [ ] Investigar reglas de negocio reales: plazos legales por tipo de proceso en Colombia, distribución de tipos de proceso por región
- [ ] Programar `src/generate_data.py` que genere 10,000 registros con:
  - **Features:** tipo_proceso, tipo_ultima_actuacion, ciudad, despacho, dias_sin_actividad, num_partes, total_actuaciones, frecuencia_actualizaciones, tiene_termino_legal, plan_suscripcion
  - **Target:** `riesgo_vencimiento` (1 = riesgo, 0 = sin riesgo)
  - **Regla de etiquetado:** riesgo = 1 si la última actuación tiene término legal Y han pasado >70% de los días hábiles sin nueva actuación
  - **Desbalance realista (~70-30)**
  - **Plan de suscripción** como feature (Freemium, Pro, Premium, Firma) con límite de procesos simulado
- [ ] Guardar en `data/raw/procesos_judiciales.csv`

### Fase 2: ETL y Feature Engineering
- [ ] Programar `src/etl.py`:
  - One-hot encoding de categóricas (tipo_proceso, tipo_actuacion, ciudad, plan)
  - Label encoding ordinal para despacho (ordenado por congestión histórica simulada)
  - Feature engineering: `pct_plazo_consumido`, `frecuencia_actualizaciones`, `dias_ultima_actuacion_log`
  - Escalamiento (StandardScaler) de numéricas
  - Split train/test (80/20) con stratify en target
- [ ] Guardar scaler y encoder en `data/processed/` para reutilizar en inferencia
- [ ] Guardar train/test sets

### Fase 3: Entrenamiento y MLflow (3 Experimentos)
- [ ] Configurar MLflow Tracking URI (local, carpeta `ml_module/mlruns/`)
- [ ] Programar `src/train.py` con 3 experimentos:

| Experimento | Modelo | Hiperparámetros | Técnica Balanceo |
| :--- | :--- | :--- | :--- |
| **Exp 1 - Baseline** | XGBoost | max_depth=4, n_estimators=100, lr=0.1 | class_weight='balanced' |
| **Exp 2 - Profundo** | XGBoost | max_depth=8, n_estimators=200, lr=0.05 | SMOTE |
| **Exp 3 - Regularizado** | XGBoost | max_depth=3, n_estimators=150, lr=0.01, reg_lambda=2 | ADASYN |

- [ ] Registrar en MLflow por experimento:
  - Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
  - Parámetros del modelo
  - Feature importance plot (artifact)
  - Matriz de confusión (artifact)
  - Curva ROC (artifact)

### Fase 4: Selección del Mejor Modelo
- [ ] Programar `src/select_best.py`:
  - Conectar a MLflow Tracking
  - Consultar runs de los 3 experimentos registrados
  - Comparar F1-Score (priorizar Recall si el F1 es similar, por ser un problema de riesgo)
  - Seleccionar el mejor run
  - Registrar en MLflow Model Registry con etiqueta "Production"
  - Exportar modelo a `data/processed/best_model.pkl` + metadata.json (features usadas, fecha, métricas)

### Fase 5: API (FastAPI)
- [ ] Programar `api/schemas.py` con modelos Pydantic:
  - `ProcesoInput`: tipo_proceso, tipo_ultima_actuacion, ciudad, despacho, dias_sin_actividad, num_partes, total_actuaciones, tiene_termino_legal, plan_suscripcion
  - `BatchInput`: lista de ProcesoInput
  - `RiskOutput`: riesgo (0/1), probabilidad, nivel (Alto/Medio/Bajo según thresholds), factores_riesgo, shap_values
- [ ] Programar `api/router.py` con endpoints:
  - `POST /predict/risk` — predicción individual con explicación SHAP
  - `POST /predict/batch` — predicción por lote (para carga masiva)
  - `GET /health` — health check + última métrica del modelo
  - `GET /model/info` — metadatos del modelo en producción (features, fecha, F1-score)
- [ ] El router está diseñado para importarse desde `backend/routers/` en producción
- [ ] Cargar modelo desde MLflow Model Registry (o fallback a `best_model.pkl`)
- [ ] Probar endpoints localmente con `uvicorn`

### Fase 6: Interfaz de Usuario (Streamlit — Prototipo Demo)
- [ ] Programar `app/streamlit_app.py` con:
  - **Pestaña 1 — Evaluación Individual:**
    - Formulario con selectboxes: tipo de proceso, tipo de actuación, ciudad, despacho, plan de suscripción
    - Input numérico: días sin actividad, número de partes, total actuaciones
    - Checkbox: tiene término legal asociado
    - Botón "Evaluar Riesgo" → llama a `POST /predict/risk`
    - Output: semáforo (🔴 Riesgo Alto / 🟡 Riesgo Medio / 🟢 Sin Riesgo), porcentaje de probabilidad
    - Desglose SHAP: barras horizontales con factores que más influyen
  - **Pestaña 2 — Carga Masiva:**
    - Subida de archivo CSV con múltiples radicados
    - Tabla de resultados con filtros por nivel de riesgo
    - Botón de exportar resultados
  - **Pestaña 3 — Dashboard de Salud:**
    - KPIs: total procesos monitoreados, en riesgo, críticos
    - Gráfico de distribución de riesgo por tipo de proceso
    - Gráfico de tendencia de riesgo en el tiempo (últimas N consultas)
    - Alertas de planes: qué procesos están cerca del límite según el plan
- [ ] Conectar con API FastAPI (configurable por variable de entorno)

### Fase 7: Tests Unitarios
- [ ] Programar `tests/test_model.py`:
  - Test de carga del modelo desde pickle
  - Test de predicción con entrada válida (verificar formato de output)
  - Test de predicción con edge cases (plan Freemium con muchos procesos, 0 días sin actividad)
  - Test de consistencia: misma entrada → misma predicción
- [ ] Programar `tests/test_api.py`:
  - Test de endpoint `/health` (status code 200)
  - Test de endpoint `/predict/risk` con datos válidos (status + schema output)
  - Test de endpoint `/predict/risk` con datos inválidos (400 error)
  - Test de endpoint `/predict/batch` con lista de 100 inputs
- [ ] Verificar que `pytest tests/` pase todos los tests

### Fase 8: Dockerización
- [ ] Crear `docker/Dockerfile.api` — imagen con FastAPI + modelo
- [ ] Crear `docker/Dockerfile.app` — imagen con Streamlit
- [ ] Crear `docker/docker-compose.yml` con servicios:
  - `api` — puerto 8000, monta volumen con modelo
  - `app` — puerto 8501, conectado a api
  - `mlflow` — MLflow Tracking Server, puerto 5000
  - `prometheus` — métricas de api, puerto 9090
  - `grafana` — dashboards, puerto 3000 (opcional)
- [ ] Probar `docker-compose up` completo localmente

### Fase 9: CI/CD (GitHub Actions)
- [ ] Crear `.github/workflows/ci.yml` en `ml_module/`:
  - Trigger: push a main + PRs
  - Jobs:
    1. **test:** Python setup, install deps, `pytest tests/`
    2. **lint:** flake8 o ruff sobre src/ y api/
    3. **docker-build:** build de imágenes Docker (no push)
- [ ] Subir a GitHub y verificar que el workflow pasa

### Fase 10: Documentación y README
- [ ] Crear `ml_module/README.md` con:
  - Descripción del problema real de Abodi y cómo el modelo lo resuelve
  - Arquitectura del módulo ML (diagrama de flujo)
  - Instrucciones de instalación (`pip install -r requirements.txt`)
  - Instrucciones de ejecución paso a paso (generar datos → entrenar → API → Streamlit)
  - Cómo integrar los endpoints al backend existente de Abodi
  - Capturas de pantalla de Streamlit funcionando
  - KPIs objetivo del modelo (F1 ≥ 0.85)
  - Enlace a demo en video

### Fase 11: Preparación de Demo y Pitch
- [ ] Preparar dataset de demostración (10 casos variados con distintos niveles de riesgo)
- [ ] Probar flujo completo: Streamlit → FastAPI → Modelo → Predicción → SHAP
- [ ] Probar demo de carga masiva con archivo CSV de 50 procesos
- [ ] Mostrar Dashboard de Salud con KPIs
- [ ] Preparar slides de Pitch (10 min):
  - Slide 1: Problema — abogados pierden 1-3h/día revisando manualmente
  - Slide 2: Solución — ML que predice vencimiento de términos automáticamente
  - Slide 3: Valor de negocio — reduce riesgo de negligencias, ahorra tiempo, escala con planes
  - Slide 4: Demo en vivo (sin código)
  - Slide 5: Resultados — F1 ≥ 0.85, integración con Abodi
- [ ] Ensayar demo en vivo de 3-5 minutos

### Fase 12: Monitoreo (deseable)
- [ ] Configurar Prometheus endpoint en FastAPI:
  - `prediction_latency_seconds` — histograma de latencia
  - `predictions_total` — contador de predicciones
  - `risk_ratio` — ratio de predicciones con riesgo > 0.5
  - `model_f1_score` — gauge con F1 del modelo en producción
- [ ] Configurar Grafana:
  - Dashboard con latencia p99, throughput, distribución de riesgo
  - Alerta si F1-score < 0.85
  - Alerta si latencia > 1s
- [ ] Simular degradación del modelo (cambiar distribución de datos) y probar alerta

---

## Dependencias Clave (requirements.txt)

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
mlflow>=2.8.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
shap>=0.43.0
pydantic>=2.5.0
pytest>=7.4.0
imbalanced-learn>=0.11.0
prometheus-client>=0.19.0
```

---

## Timeline Estimado

| Fase | Horas | Depende de |
| :--- | :--- | :--- |
| Fase 0: Setup | 1h | — |
| Fase 1: Datos sintéticos | 2h | Fase 0 |
| Fase 2: ETL | 1h | Fase 1 |
| Fase 3: MLflow (3 exp) | 2h | Fase 2 |
| Fase 4: Selección | 1h | Fase 3 |
| Fase 5: FastAPI | 2h | Fase 4 |
| Fase 6: Streamlit | 3h | Fase 5 |
| Fase 7: Tests | 1h | Fase 5,6 |
| Fase 8: Docker | 2h | Fase 5,6 |
| Fase 9: CI/CD | 1h | Fase 7,8 |
| Fase 10: README | 1h | Todo |
| Fase 11: Demo | 2h | Todo |
| Fase 12: Monitoreo | 2h (opcional) | Fase 5,6 |
| **Total** | **21h (~3 días)** | — |

---

## Verificación Final (Demo Day)

- [ ] `python src/generate_data.py` — genera 10,000 registros sintéticos realistas
- [ ] `python src/etl.py` — transforma, guarda scaler y split
- [ ] `python src/train.py` — corre 3 experimentos y los registra en MLflow
- [ ] `python src/select_best.py` — selecciona mejor modelo, lo registra en Model Registry
- [ ] `mlflow ui` — muestra experimentos con métricas y artifacts
- [ ] `pytest tests/` — todos los tests pasan
- [ ] `uvicorn api.router:app` — endpoints responden correctamente
- [ ] `streamlit run app/streamlit_app.py` — UI funcional con semáforo, carga masiva y dashboard de salud
- [ ] `docker-compose up` — api + app + mlflow se levantan sin errores
- [ ] GitHub Actions pipeline en verde
- [ ] Demo en vivo: formulario → predicción → SHAP → alerta funciona E2E
- [ ] Presentación sin código, enfocada en problema/solución/valor de negocio

---

*Creado para el proyecto final de MLOps - Abodi App.*
