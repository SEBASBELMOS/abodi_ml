# Propuestas de Modelos de Machine Learning para Abodi App (MLOps)

Este documento detalla cuatro propuestas de modelos de Machine Learning para la plataforma **Abodi App**, ordenadas de **menor a mayor complejidad**. Se priorizan modelos de **clasificación** porque el checklist del proyecto exige Accuracy, F1-Score y Recall — métricas propias de clasificación, no de regresión.

---

## Tabla Comparativa de Complejidad

| Propuesta | Tipo de Modelo | Tipo de Datos | Facilidad | Tiempo | MLOps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Duración (clasificación de plazos)** | Clasificación Multiclase | Numérico / Categórico | Alta | 1-2 días | XGBoost, Random Forest |
| **2. Vencimiento de Términos** | Clasificación Binaria | Numérico / Categórico | Alta | 1-2 días | XGBoost + SHAP |
| **3. Éxito de Tutelas** | Clasificación Binaria | Categórico | Media | 2-3 días | XGBoost + SHAP |
| **4. Riesgo de Pérdida del Proceso** | Clasificación Multiclase | Numérico / Categórico | Media | 2-3 días | XGBoost + SHAP |

---

## Opción 1: Predictor de Duración con Clasificación de Plazos (Fácil)

Estima en qué rango de tiempo se resolverá un proceso judicial. **Versión clasificación** para alinearse con las métricas del proyecto.

### Por qué es fácil:
1. **Datos tabulares simples** — sin NLP.
2. **Generación sintética trivial** — tutelas 10-15 días, ordinarios 300-800 días, Bogotá más lento que Tunja.
3. **Entrenamiento veloz** — XGBoost/Random Forest en segundos.
4. **UI sencilla** — solo selectboxes + radicado.

### Ficha Técnica:
- **Tipo de Tarea:** Clasificación Multiclase
- **Target:** `Corto (< 90 días)` / `Medio (90-365 días)` / `Largo (> 365 días)`
- **Features:**
  - Ciudad / Departamento (extraído del radicado, dígitos 1-5)
  - Especialidad / Despacho (dígitos 6-9 del radicado)
  - Tipo de Proceso (Ejecutivo, Ordinario, Tutela, Laboral)
  - Año de Radicación
  - Cuantía del Proceso (baja / media / alta)
- **Métricas:** Accuracy, F1-score (macro/weighted), Recall, Matriz de Confusión
- **MLflow:** 3 experimentos variando max_depth, n_estimators, learning_rate
- **Visualización Streamlit:** Gráfico de barras con probabilidad por categoría + indicador de congestión del despacho vs promedio nacional

### Checklist MLOps:
- [x] Accuracy, F1, Recall
- [x] MLflow con 2-3 experimentos variando hiperparámetros
- [x] Mejor modelo registrado en MLflow Model Registry
- [x] FastAPI endpoint: `POST /predict/duration`
- [x] Streamlit con inputs y predicción en vivo

---

## Opción 2: Predictor de Vencimiento de Términos (Fácil - RECOMENDADA)

Predice si un proceso judicial está en riesgo inminente de que se venza un término procesal. Este es el **problema central que resuelve Abodi App**: evitar que los abogados pierdan plazos por no monitorear manualmente.

### Por qué es la mejor opción:
1. **Máximo alineamiento con el negocio** — Resuelve el dolor principal de los abogados y es literalmente el value proposition de Abodi.
2. **Clasificación binaria pura** — Ideal para Accuracy, F1, Recall, Matriz de Confusión.
3. **Features tabulares simples** — Sin NLP.
4. **Demostrable en vivo** — El usuario ingresa datos de un caso y ve si "requiere acción urgente" o "está al día".
5. **Monitoreo con sentido** — La deriva del modelo es real: si la Rama Judicial cambia sus plazos, las predicciones se degradan. Perfecto para Grafana/Prometheus.

### Ficha Técnica:
- **Tipo de Tarea:** Clasificación Binaria
- **Target:** `1 - Riesgo de vencimiento` / `0 - Sin riesgo`
- **Features:**
  - Días desde la última actuación registrada
  - Tipo de proceso (Tutela, Ordinario, Ejecutivo, Laboral)
  - Tipo de última actuación (Auto interlocutorio, Sentencia, Notificación, Traslado)
  - Despacho / Juzgado (categorizado por congestión histórica)
  - Ciudad (Bogotá, Medellín, Cali, Otras)
  - Tiene término legal asociado la última actuación (Sí/No)
  - Cantidad de partes en el proceso
- **Generación sintética:** Las tutelas tienen plazos de 3-10 días hábiles para responder; los ordinarios tienen plazos más largos (10-30 días). Se simula riesgo cuando la última actuación tiene término y han pasado >70% de los días hábiles sin nueva actuación.
- **Métricas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Matriz de Confusión
- **Desbalance de clases:** Aprox 70-30 sin riesgo vs con riesgo. Aplicar SMOTE o class_weight.
- **MLflow:** 3 experimentos con distintas proporciones de sobremuestreo y parámetros de XGBoost
- **Visualización Streamlit:** Dashboard con semáforo (rojo = riesgo, verde = ok), timeline de actuaciones, y alertas de términos

### Checklist MLOps:
- [x] Accuracy, F1, Recall, Matriz de Confusión
- [x] MLflow con experimentos variando scale_pos_weight, max_depth
- [x] Model Registry con la mejor versión etiquetada "Production"
- [x] FastAPI: `POST /predict/risk` y `POST /predict/batch`
- [x] Streamlit: formulario de entrada + panel de riesgo + historial
- [x] Monitoreo: drift detection en distribución de "días sin actividad" (ideal para Prometheus)

---

## Opción 3: Predictor de Éxito en Acciones de Tutela (Media)

Predice la probabilidad de que una acción de tutela sea concedida (a favor del accionante) o negada/improcedente.

### Por qué es de dificultad media:
1. **Modelado tabular** — similar a Opción 1, datos categóricos estructurados.
2. **Desbalance de clases** — en la realidad, la mayoría de tutelas se niegan o declaran improcedentes. Requiere balanceo.
3. **Métricas de evaluación** — priorizar Recall y F1-Score, no solo Accuracy.
4. **Explicabilidad valiosa** — SHAP para mostrar qué factores pesaron en la decisión. Ideal para un abogado que quiere entender el "por qué".

### Ficha Técnica:
- **Tipo de Tarea:** Clasificación Binaria
- **Target:** `1 - Concedida / Éxito` / `0 - Negada / Improcedente`
- **Features:**
  - Derecho Fundamental invocado (Salud, Petición, Debido Proceso, Mínimo Vital, Libre Desarrollo de la Personalidad)
  - Entidad Accionada (Nueva EPS, Colpensiones, Alcaldía Municipal, Fiscalía, Secretaría de Educación)
  - Tipo de Accionante (Persona Natural / Persona Jurídica / Entidad Pública)
  - Ciudad / Departamento del juzgado
  - Juzgado específico (algunos jueces son más garantistas que otros)
  - Año de radicación (cambios jurisprudenciales)
  - Tiene medida provisional solicitada (Sí/No)
- **Métricas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Matriz de Confusión
- **MLflow:** 3 experimentos variando técnica de balanceo (SMOTE, ADASYN, class_weight) y parámetros del modelo
- **Explicabilidad:** SHAP summary plot + force plot para predicciones individuales
- **Visualización Streamlit:** Medidor de probabilidad circular, desglose de factores de riesgo estilo "semáforo" por feature, y análisis SHAP

### Checklist MLOps:
- [x] Accuracy, F1, Recall, Matriz de Confusión
- [x] MLflow con experimentos variando balancing + hiperparámetros
- [x] Model Registry con mejor versión
- [x] FastAPI: `POST /predict/tutela`
- [x] Streamlit: formulario de tutela + resultado explicado con SHAP
- [x] Monitoreo: seguimiento de distribución de derechos invocados (drift)

---

## Opción 4: Predictor de Riesgo de Pérdida del Proceso (Media)

Clasifica un proceso judicial en niveles de riesgo de ser fallado en contra del cliente. Valor estratégico alto para firmas legales.

### Por qué es de dificultad media:
1. **Features derivadas** — requiere ingeniería de características como "ratio de actuaciones a favor/en contra", que debe calcularse durante la generación sintética.
2. **Multiclase con orden** — las categorías tienen orden intrínseco (Bajo < Medio < Alto), lo que permite usar ordinal classification o simplemente multiclase.
3. **SHAP interpretable** — alto valor para el abogado: "¿por qué este caso es de alto riesgo?"

### Ficha Técnica:
- **Tipo de Tarea:** Clasificación Multiclase (ordinal)
- **Target:** `Bajo` / `Medio` / `Alto`
- **Features:**
  - Tipo de Proceso (Ejecutivo, Ordinario, Tutela, Laboral, Administrativo)
  - Calidad del Demandante / Demandado (Persona Natural, Empresa, Entidad Pública)
  - Cuantía del Proceso
  - Despacho / Juzgado (tasa histórica de fallos en contra)
  - Ciudad
  - Número de actuaciones registradas (más actuaciones = mayor complejidad)
  - Ratio de actuaciones a favor vs en contra
  - Duración acumulada del proceso en días
  - Tiene sentencia de primera instancia (Sí/No)
  - En etapa de apelación (Sí/No)
- **Métricas:** Accuracy, F1-Score (macro), Recall por clase, Matriz de Confusión, Coeficiente Kappa
- **MLflow:** 3 experimentos con diferentes codificaciones de features y arquitecturas (XGBoost multiclase vs OneVsRest)
- **Visualización Streamlit:** Dashboard de caso con medidor de riesgo (bajo/medio/alto), factores contributing (SHAP horizontal bar), y recomendaciones accionables

### Checklist MLOps:
- [x] Accuracy, F1, Recall, Matriz de Confusión
- [x] MLflow con experimentos variando codificación de features
- [x] Model Registry
- [x] FastAPI: `POST /predict/risk-level`
- [x] Streamlit: input form + risk dashboard + SHAP explanations
- [x] Monitoreo: drift en distribución de tipos de proceso y cuantías

---

## Tabla Comparativa vs Checklist del Proyecto

| Requisito MLOps | Opción 1 (Duración) | Opción 2 (Términos) | Opción 3 (Tutelas) | Opción 4 (Riesgo) |
| :--- | :--- | :--- | :--- | :--- |
| Clasificación (Acc, F1, Recall) | Multiclase | Binaria | Binaria | Multiclase |
| MLflow (2-3 experimentos) | max_depth, n_estimators | scale_pos_weight, max_depth | SMOTE vs ADASYN params | Encoding + params |
| Model Registry | Sí | Sí | Sí | Sí |
| FastAPI endpoint | 1 endpoint | 2 endpoints | 1 endpoint | 1 endpoint |
| Streamlit UI | Selectboxes + gráfico | Formulario + semáforo | Formulario + SHAP | Dashboard + SHAP |
| Datos sintéticos | Muy fácil | Fácil | Fácil | Media |
| Dockerización | Trivial | Trivial | Trivial | Trivial |
| Monitoreo (drift) | Bajo | **Alto** (relevante) | Medio | Medio |

---

## Recomendación Final

### Si priorizas entregar rápido y sin riesgo técnico: **Opción 1 (Duración - Clasificación)**
- Datos sintéticos más simples de generar
- UI con pocos componentes
- 3 experimentos fáciles de variar
- Riesgo técnico casi nulo

### Si priorizas impacto real + alineación con el negocio: **Opción 2 (Vencimiento de Términos)** ← MI RECOMENDACIÓN PRINCIPAL
- Resuelve el problema central de Abodi
- Clasificación binaria perfecta para métricas del proyecto
- Features fáciles de generar sintéticamente
- Demo impactante: el usuario ve si su caso "está en riesgo" con semáforo
- Monitoreo con sentido real: la distribución de "días sin actuación" cambia con el tiempo
- SHAP añade explicabilidad sin mucha complejidad extra

### Si quieres destacar con algo más estratégico: **Opción 4 (Riesgo de Pérdida)**
- Valor diferenciador frente a otras herramientas del mercado
- Permite mostrar análisis SHAP avanzado
- Buen equilibrio entre complejidad e impacto

> **Conclusión:** La Opción 2 (Vencimiento de Términos) es el sweet spot: fácil como la Opción 1, pero con el impacto de negocio de las más complejas. Te permite concentrar tiempo en MLflow + FastAPI + Streamlit + Docker, que es lo que realmente se evalúa.

---

## Anexo: Propuesta Avanzada (NLP) — No recomendada para el plazo actual

Esta propuesta quedó fuera de las 4 principales por su alta complejidad, pero se documenta como referencia.

**Clasificador de Urgencia por NLP:** Analiza el texto de la última actuación judicial para clasificar urgencia y calcular términos legales.

**Por qué NO se recomienda para este proyecto:**
1. Generar texto sintético jurídico en español que suene natural es muy difícil
2. El modelo (BETO o transformers) pesa >500MB, complica Docker
3. Entrenamiento lento incluso en GPU
4. 3-4 días de desarrollo vs 1-2 de las opciones tabulares
5. El riesgo de no terminar a tiempo es alto

Para una versión 2.0 del proyecto, sería el siguiente paso natural.

---

*Actualizado para el proyecto final de MLOps - Abodi App.*
