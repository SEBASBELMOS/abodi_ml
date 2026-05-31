# `context.md` - Proyecto Final: Implementación de Metodologías MLOps

## 1. Introducción y Propósito

### Contexto del Producto Real
**Abodi 2.0** es una plataforma SaaS colombiana de monitoreo judicial automatizado. Su función principal es centralizar, automatizar y optimizar el monitoreo de procesos judiciales para abogados, firmas legales y departamentos jurídicos.

**Problema real:** Los abogados en Colombia pierden de 1 a 3 horas diarias revisando manualmente los portales de la Rama Judicial, con un alto riesgo de omitir actuaciones legales críticas y vencimiento de términos.

**Misión:** Empoderar a abogados mediante la automatización que elimine tareas administrativas repetitivas y reduzca riesgos procesales.

### Propósito del Proyecto MLOps
Trascender la fase experimental de la IA (basada únicamente en notebooks) y avanzar hacia un enfoque **End-to-End (E2E)**. Implementar un modelo de Machine Learning escalable, monitoreable y listo para producción que se integre a la plataforma Abodi 2.0 existente.

## 2. Metodología: MLOps y Ciclo de Vida
El proyecto integra la intersección entre **Machine Learning, Data Engineering y DevOps**. Se enfatiza la observabilidad, ya que los modelos se degradan con el tiempo debido a que los datos del mundo cambian (ejemplo de la marca Zara y sus precios desactualizados).

## 3. Arquitectura del Sistema y Herramientas

### Stack Tecnológico Real de Abodi 2.0
| Componente | Tecnología |
| :--- | :--- |
| Frontend SaaS | **Vue 3 + Vite** |
| Backend API | **FastAPI** |
| Base de Datos | **Supabase (PostgreSQL)** — 28+ tablas |
| Infraestructura | **DigitalOcean** (VPS) |
| Pasarela de Pagos | **Wompi** |
| Email / Notificaciones | **Resend** |
| Proxies para Scraping | **Webshare** |
| Autenticación | **Supabase Auth** |

### Componentes del Proyecto MLOps (Este Entregable)
El flujo de trabajo del modelo ML debe seguir estos componentes técnicos:
* **Registro de Experimentos (MLflow):** Seguimiento de métricas (Accuracy, F1-score, Recall) y registro automático para seleccionar siempre el mejor modelo para producción.
* **Integración Continua (CI/CD):** Uso de **GitHub Actions** para pruebas unitarias, verificación de dependencias y aseguramiento del código antes de dockerizar o desplegar.
* **Servicio de Inferencia (API):** Endpoints en **FastAPI** que consuman el mejor modelo registrado. Estos endpoints se integran al backend existente de Abodi.
* **Interfaz de Demostración (App):** Prototipo funcional en **Streamlit** que consuma la API del modelo para demostración y pruebas.
* **Monitoreo y Alertas:** Uso de **Grafana/Prometheus** para visualizar latencia, inferencias y activar alarmas si el rendimiento cae por debajo de un umbral (ej. 0.85).

### Funcionalidades Clave del Producto
* **Sincronización Automática:** Consultas a la Rama Judicial 4 veces al día (8AM, 11AM, 2PM, 5PM).
* **Búsqueda Inteligente:** Consulta de procesos por número de radicación de 23 dígitos directamente desde la base de datos oficial.
* **Notificaciones en Tiempo Real:** Alertas vía **WebSockets** y correos electrónicos ante cambios de estado o nuevas actuaciones.
* **Planes de Suscripción:** Freemium, Pro, Premium y Firma, con límites de procesos por plan.
* **Carga Masiva:** Subida simultánea de múltiples radicados.
* **Dashboard de Salud:** Vista unificada con procesos atrasados, críticos y al día.

## 4. Estructura de Carpetas

### Estructura Real del Repositorio Abodi 2.0
```
abodi/
├── backend/           # FastAPI (lógica de negocio y endpoints)
├── frontend/          # Vue 3 + Vite (interfaz SaaS)
├── scripts/           # Scrapers y utilidades de automatización
├── test_files/        # Archivos para pruebas de carga y validación
├── supabase/          # Esquemas y migraciones de BD
└── docker/            # Configuración de contenedores
```

### Estructura del Módulo ML (Este Entregable)
Dentro del repositorio, el módulo de ML sigue esta estructura estándar para el proyecto:

```
ml_module/
├── api/               # Endpoints FastAPI para el modelo (se integra a backend/)
├── app/               # Streamlit (prototipo de demostración)
├── data/              # Datos sintéticos (raw/ y processed/)
├── src/               # Scripts de entrenamiento, ETL y selección
├── tests/             # Tests unitarios para CI/CD
├── docker/            # Dockerfiles para el módulo ML
├── requirements.txt   # Dependencias del módulo ML
└── README.md          # Documentación del módulo
```

## 5. Métricas de Negocio y Operativas (KPIs)
El monitoreo del sistema debe medir estos objetivos reales del producto:

| KPI | Meta |
| :--- | :--- |
| **Uptime de Sincronización** | ≥ 99.5% |
| **Cobertura de Juzgados** (scraping) | ≥ 95% del país |
| **Latencia de Notificación** (actuación → alerta) | < 4 horas |
| **Rendimiento del Modelo (F1-Score)** | ≥ 0.85 |
| **Precisión en Predicción de Riesgo** | ≥ 80% |

---

## ✅ Checklist de Entrega Final (Lunes - "Demo Day")

### A. Desarrollo Técnico y Prototipo
- [x] **Modelo y MLflow:** Registro de 3 experimentos (Baseline, SMOTE, ADASYN) con variación de hiperparámetros. Mejor modelo: ADASYN + XGBoost (F1: 0.843, AUC: 0.905). Registrado en MLflow Model Registry.
- [x] **API y App Funcional:** FastAPI con 5 endpoints operativos. Streamlit con 3 pestañas (evaluación individual, carga masiva, dashboard de salud). 13 tests unitarios pasando.
- [x] **Integración con Backend Existente:** Los endpoints están diseñados como router de FastAPI importable desde `backend/routers/`.
- [x] **Plan de Suscripción:** El modelo incluye `plan_suscripcion` como feature. La UI muestra límites por plan (Freemium: 5, Pro: 50, Premium: 200, Firma: ilimitado).
- [x] **Carga Masiva:** Streamlit permite subir CSV con múltiples procesos y evaluarlos por lote vía `POST /predict/batch`.
- [x] **Dashboard de Salud:** Vista unificada con KPIs, distribución de riesgo, límites por plan y tabla de procesos de ejemplo.
- [ ] **Repositorio GitHub:** Pendiente de subir. Código completo en `ml_module/` listo para push.

### B. Presentación (Pitch de 10 Minutos)
- [ ] **Enfoque Comercial:** El Pitch debe centrarse en el Problema, la Solución y el Valor de Negocio.
- [ ] **Demo en Vivo:** Mostrar el producto funcionando en tiempo real.
- [ ] **Restricción:** No se debe mostrar código durante esta presentación; el objetivo es vender la solución como producto o servicio.
- [ ] **Participantes:** Máximo 1 o 2 personas exponen el pitch, aunque todo el equipo puede responder preguntas.

### C. Artículo Científico (Versión Preliminar)
- [ ] **Formato:** Máximo 6 páginas utilizando Overleaf (LaTeX).
- [ ] **Contenido:** Introducción, estado del arte, descripción de la solución y resultados preliminares.
- [ ] **Plantilla:** Se debe utilizar la plantilla de Overleaf proporcionada por el profesor.

---

> **Nota del profesor:** *"No se queden solo con que el modelo funciona; lo importante es el mantenimiento, la producción y el monitoreo. El mundo cambia y los modelos se degradan."*
