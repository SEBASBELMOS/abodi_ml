import io
import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Abodi ML | Riesgo de terminos",
    page_icon="AB",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

TIPO_PROCESO_OPTIONS = [
    "Tutela",
    "Ordinario Civil",
    "Ejecutivo",
    "Laboral",
    "Administrativo",
]

TIPO_ACTUACION_OPTIONS = [
    "Auto admisorio demanda",
    "Traslado excepciones",
    "Traslado recurso",
    "Sentencia primera instancia",
    "Notificaci\u00c3\u00b3n por estado",
    "Auto de pruebas",
    "Fijaci\u00c3\u00b3n audiencia",
    "Providencia interlocutoria",
    "Constancia secretarial",
    "Oficio comisorio",
]

CIUDAD_OPTIONS = [
    "Bogot\u00c3\u00a1",
    "Medell\u00c3\u00adn",
    "Cali",
    "Barranquilla",
    "Manizales",
    "Bucaramanga",
    "Otras",
]

DESPACHO_OPTIONS = [
    "Juzgado Civil Municipal",
    "Juzgado Penal Municipal",
    "Juzgado de Familia",
    "Juzgado Civil del Circuito",
    "Juzgado Laboral Municipal",
    "Juzgado Laboral del Circuito",
    "Juzgado Administrativo",
    "Tribunal Administrativo",
]

PLAN_OPTIONS = ["Freemium", "Pro", "Premium", "Firma"]

PLAN_LIMITS = {
    "Freemium": 5,
    "Pro": 50,
    "Premium": 200,
    "Firma": "Ilimitado",
}

DISPLAY_LABELS = {
    "Notificaci\u00c3\u00b3n por estado": "Notificacion por estado",
    "Fijaci\u00c3\u00b3n audiencia": "Fijacion audiencia",
    "Bogot\u00c3\u00a1": "Bogota",
    "Medell\u00c3\u00adn": "Medellin",
}

INPUT_ALIASES = {
    "Notificacion por estado": "Notificaci\u00c3\u00b3n por estado",
    "Notificaci\u00f3n por estado": "Notificaci\u00c3\u00b3n por estado",
    "Fijacion audiencia": "Fijaci\u00c3\u00b3n audiencia",
    "Fijaci\u00f3n audiencia": "Fijaci\u00c3\u00b3n audiencia",
    "Bogota": "Bogot\u00c3\u00a1",
    "Bogot\u00e1": "Bogot\u00c3\u00a1",
    "Medellin": "Medell\u00c3\u00adn",
    "Medell\u00edn": "Medell\u00c3\u00adn",
}

LEVEL_STYLES = {
    "Alto": {"color": "#dc2626", "bg": "#fef2f2", "label": "Riesgo alto"},
    "Medio": {"color": "#b45309", "bg": "#fffbeb", "label": "Riesgo medio"},
    "Bajo": {"color": "#047857", "bg": "#ecfdf5", "label": "Riesgo bajo"},
    "Error": {"color": "#475569", "bg": "#f8fafc", "label": "Error"},
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.sidebar.toggle("Modo oscuro", key="dark_mode")

if dark_mode:
    LEVEL_STYLES = {
        "Alto": {"color": "#f87171", "bg": "#3b1117", "label": "Riesgo alto"},
        "Medio": {"color": "#fbbf24", "bg": "#33220b", "label": "Riesgo medio"},
        "Bajo": {"color": "#34d399", "bg": "#082f24", "label": "Riesgo bajo"},
        "Error": {"color": "#cbd5e1", "bg": "#111827", "label": "Error"},
    }


st.markdown(
    """
    <style>
    :root {
      --abodi-ink: #172033;
      --abodi-muted: #64748b;
      --abodi-line: #dbe3ef;
      --abodi-panel: #ffffff;
      --abodi-soft: #f6f8fb;
      --abodi-blue: #2563eb;
      --abodi-teal: #0f766e;
    }

    .main .block-container {
      padding-top: 1.4rem;
      padding-bottom: 3rem;
      max-width: 1220px;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
      background: #f6f8fb;
      color: var(--abodi-ink);
    }

    [data-testid="stHeader"] {
      border-bottom: 1px solid rgba(219, 227, 239, 0.8);
    }

    [data-testid="stSidebar"] {
      background: #f8fafc;
      border-right: 1px solid var(--abodi-line);
    }

    [data-testid="stSidebar"] * {
      color: var(--abodi-ink);
    }

    [data-testid="stSidebar"] code {
      color: #e2e8f0;
    }

    h1, h2, h3 {
      color: var(--abodi-ink);
      letter-spacing: 0;
    }

    p, li, label, span, div {
      letter-spacing: 0;
    }

    label,
    label p,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
      color: #334155 !important;
      font-weight: 700;
    }

    div[data-testid="stForm"] {
      border: 1px solid var(--abodi-line);
      background: #ffffff;
      border-radius: 8px;
      padding: 18px 18px 8px 18px;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
      background: #ffffff;
      border-color: #cbd5e1;
      color: var(--abodi-ink);
    }

    div[data-baseweb="select"] span,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
      color: var(--abodi-ink);
    }

    div[data-testid="stDataFrame"] {
      border-radius: 8px;
      overflow: hidden;
    }

    .hero {
      border: 1px solid var(--abodi-line);
      background: linear-gradient(135deg, #ffffff 0%, #f7fbff 62%, #eefcf9 100%);
      border-radius: 8px;
      padding: 24px 26px;
      margin-bottom: 18px;
    }

    .hero-kicker {
      color: var(--abodi-teal);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .hero-title {
      color: var(--abodi-ink);
      font-size: 2.1rem;
      font-weight: 800;
      line-height: 1.1;
      margin: 0 0 8px 0;
    }

    .hero-copy {
      color: #475569;
      font-size: 1rem;
      line-height: 1.55;
      max-width: 760px;
      margin: 0;
    }

    .status-row {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin: 14px 0 18px 0;
    }

    .metric-card {
      border: 1px solid var(--abodi-line);
      background: var(--abodi-panel);
      border-radius: 8px;
      padding: 15px 16px;
    }

    .metric-label {
      color: var(--abodi-muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 5px;
    }

    .metric-value {
      color: var(--abodi-ink);
      font-size: 1.45rem;
      font-weight: 800;
      line-height: 1.2;
    }

    .metric-note {
      color: var(--abodi-muted);
      font-size: 0.82rem;
      margin-top: 4px;
    }

    .section-title {
      color: var(--abodi-ink);
      font-size: 1.2rem;
      font-weight: 800;
      margin: 8px 0 4px 0;
    }

    .section-subtitle {
      color: var(--abodi-muted);
      font-size: 0.92rem;
      margin: 0 0 14px 0;
    }

    .result-card {
      border: 1px solid var(--abodi-line);
      background: var(--abodi-panel);
      border-radius: 8px;
      padding: 20px;
      min-height: 310px;
    }

    .risk-panel {
      border: 1px solid;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
    }

    .risk-label {
      font-size: 0.86rem;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .risk-probability {
      font-size: 3.2rem;
      line-height: 1;
      font-weight: 900;
      margin: 4px 0 8px;
    }

    .risk-caption {
      color: #475569;
      font-size: 0.9rem;
      margin: 0;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 800;
      border: 1px solid currentColor;
    }

    .sidebar-brand {
      color: var(--abodi-ink);
      font-size: 1.45rem;
      font-weight: 900;
      margin-bottom: 2px;
    }

    .sidebar-caption {
      color: var(--abodi-muted);
      font-size: 0.9rem;
      line-height: 1.45;
      margin-bottom: 14px;
    }

    .stButton > button {
      border-radius: 6px;
      border: 1px solid #1d4ed8;
      background: #2563eb;
      color: #ffffff;
      font-weight: 800;
      min-height: 42px;
    }

    .stDownloadButton > button {
      border-radius: 6px;
      font-weight: 700;
      min-height: 40px;
    }

    div[data-testid="stTabs"] button {
      font-weight: 750;
    }

    @media (max-width: 900px) {
      .status-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .hero-title {
        font-size: 1.65rem;
      }
    }

    @media (max-width: 620px) {
      .status-row {
        grid-template-columns: 1fr;
      }
      .hero {
        padding: 18px;
      }
      .risk-probability {
        font-size: 2.4rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if dark_mode:
    st.markdown(
        """
        <style>
        :root {
          --abodi-ink: #e5eefb;
          --abodi-muted: #9aa8bd;
          --abodi-line: #253247;
          --abodi-panel: #111827;
          --abodi-soft: #0b1120;
          --abodi-blue: #60a5fa;
          --abodi-teal: #5eead4;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
          background: #0b1120 !important;
          color: var(--abodi-ink) !important;
        }

        [data-testid="stHeader"] {
          border-bottom: 1px solid rgba(37, 50, 71, 0.9) !important;
        }

        [data-testid="stSidebar"] {
          background: #0f172a !important;
          border-right: 1px solid var(--abodi-line) !important;
        }

        [data-testid="stSidebar"] * {
          color: var(--abodi-ink) !important;
        }

        [data-testid="stSidebar"] code,
        [data-testid="stCodeBlock"] {
          background: #020617 !important;
          color: #dbeafe !important;
        }

        .hero {
          background: linear-gradient(135deg, #111827 0%, #0f1f34 62%, #082f2b 100%) !important;
          border-color: var(--abodi-line) !important;
        }

        .hero-copy,
        .risk-caption,
        .section-subtitle,
        .metric-note,
        .sidebar-caption {
          color: var(--abodi-muted) !important;
        }

        .metric-card,
        .result-card,
        div[data-testid="stForm"] {
          background: var(--abodi-panel) !important;
          border-color: var(--abodi-line) !important;
        }

        label,
        label p,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
          color: #cbd5e1 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stFileUploader"] section {
          background: #0f172a !important;
          border-color: #334155 !important;
          color: var(--abodi-ink) !important;
        }

        div[data-baseweb="select"] span,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
          color: var(--abodi-ink) !important;
        }

        div[data-testid="stTabs"] button {
          color: #cbd5e1 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
          color: #f8fafc !important;
        }

        .stButton > button {
          background: #3b82f6 !important;
          border-color: #60a5fa !important;
          color: #ffffff !important;
        }

        .stDownloadButton > button {
          background: #111827 !important;
          border-color: #334155 !important;
          color: #dbeafe !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def status_pill(text: str, color: str, bg: str) -> str:
    return f"""
    <span class="pill" style="color:{color}; background:{bg};">{text}</span>
    """


def level_style(level: str) -> dict:
    return LEVEL_STYLES.get(level, LEVEL_STYLES["Error"])


def render_header(health: dict | None) -> None:
    status = "Conectada" if health else "Sin conexion"
    status_color = "#047857" if health else "#dc2626"
    status_bg = "#ecfdf5" if health else "#fef2f2"
    f1_score = health.get("model_f1") if health else None
    total_predictions = health.get("total_predictions") if health else 0

    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">Abodi ML</div>
          <h1 class="hero-title">Priorizacion inteligente de procesos judiciales</h1>
          <p class="hero-copy">
            Evalua el riesgo de vencimiento de terminos con un modelo XGBoost servido por FastAPI.
            La demo esta pensada para mostrar decisiones accionables: que proceso atender primero,
            que tan urgente es y que factores explican la prediccion.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns(4)
    cards = [
        metric_card("Estado API", status_pill(status, status_color, status_bg), API_URL),
        metric_card("Modelo", "XGBoost", "Clasificador binario"),
        metric_card("F1 en produccion", f"{f1_score:.4f}" if f1_score else "N/D", "Objetivo: 0.8500"),
        metric_card("Predicciones", str(total_predictions), "Sesion actual de la API"),
    ]
    for column, card in zip(status_cols, cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)


def build_payload(
    tipo_proceso: str,
    tipo_actuacion: str,
    ciudad: str,
    despacho: str,
    dias: int,
    num_partes: int,
    total_acts: int,
    frecuencia: float,
    tiene_termino: bool,
    plan: str,
) -> dict:
    return {
        "tipo_proceso": tipo_proceso,
        "tipo_ultima_actuacion": tipo_actuacion,
        "ciudad": ciudad,
        "despacho": despacho,
        "dias_sin_actividad": dias,
        "num_partes": num_partes,
        "total_actuaciones": total_acts,
        "frecuencia_actualizaciones": frecuencia,
        "tiene_termino_legal": 1 if tiene_termino else 0,
        "plan_suscripcion": plan,
    }


def display_label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value)


def normalize_value(value):
    if isinstance(value, str):
        return INPUT_ALIASES.get(value, value)
    return value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ["tipo_ultima_actuacion", "ciudad"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_value)
    return normalized


def render_risk_result(result: dict, payload: dict) -> None:
    level = result.get("nivel", "Error")
    probability = float(result.get("probabilidad", 0.0))
    style = level_style(level)

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="risk-panel" style="border-color:{style['color']}; background:{style['bg']};">
              <div class="risk-label" style="color:{style['color']};">{style['label']}</div>
              <div class="risk-probability" style="color:{style['color']};">{probability:.1%}</div>
              <p class="risk-caption">Probabilidad estimada de vencimiento de terminos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Resumen del caso")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Campo": "Tipo de proceso", "Valor": payload["tipo_proceso"]},
                    {"Campo": "Ultima actuacion", "Valor": display_label(payload["tipo_ultima_actuacion"])},
                    {"Campo": "Ciudad", "Valor": display_label(payload["ciudad"])},
                    {"Campo": "Despacho", "Valor": payload["despacho"]},
                    {"Campo": "Dias sin actividad", "Valor": payload["dias_sin_actividad"]},
                    {"Campo": "Plan", "Valor": payload["plan_suscripcion"]},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown("#### Factores de decision")
        factors = result.get("factores_riesgo", [])
        if factors:
            factors_df = pd.DataFrame(factors)
            factors_df["impact"] = factors_df["impact"].astype(float).round(4)
            factors_df["direction"] = factors_df["direction"].replace(
                {
                    "aumenta_riesgo": "Aumenta riesgo",
                    "disminuye_riesgo": "Disminuye riesgo",
                    "explicacion_shap": "Explicacion SHAP",
                }
            )
            factors_df = factors_df.rename(
                columns={
                    "feature": "Variable",
                    "impact": "Impacto",
                    "direction": "Direccion",
                }
            )
            st.dataframe(factors_df, hide_index=True, use_container_width=True)
        else:
            st.info("La prediccion no retorno factores detallados para este caso.")

        if level == "Alto":
            st.error("Accion recomendada: revisar el proceso de inmediato y validar el termino asociado.")
        elif level == "Medio":
            st.warning("Accion recomendada: revisar el proceso durante la jornada y confirmar novedades.")
        else:
            st.success("Accion recomendada: mantener monitoreo normal.")


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tipo_proceso": ["Tutela", "Ordinario Civil", "Ejecutivo", "Laboral"],
            "tipo_ultima_actuacion": [
                "Notificaci\u00c3\u00b3n por estado",
                "Auto admisorio demanda",
                "Traslado excepciones",
                "Fijaci\u00c3\u00b3n audiencia",
            ],
            "ciudad": ["Bogot\u00c3\u00a1", "Medell\u00c3\u00adn", "Cali", "Bucaramanga"],
            "despacho": [
                "Juzgado Civil Municipal",
                "Juzgado Civil del Circuito",
                "Juzgado Civil Municipal",
                "Juzgado Laboral del Circuito",
            ],
            "dias_sin_actividad": [15, 45, 8, 21],
            "num_partes": [2, 3, 4, 2],
            "total_actuaciones": [3, 12, 7, 9],
            "frecuencia_actualizaciones": [1.5, 2.0, 1.2, 1.8],
            "tiene_termino_legal": [1, 1, 1, 1],
            "plan_suscripcion": ["Pro", "Premium", "Freemium", "Firma"],
        }
    )


def run_batch_prediction(df: pd.DataFrame) -> list[dict]:
    procesos = normalize_dataframe(df).to_dict(orient="records")
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_size = 50

    for start in range(0, len(procesos), batch_size):
        batch = procesos[start : start + batch_size]
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"procesos": batch},
            timeout=30,
        )
        response.raise_for_status()
        all_results.extend(response.json()["resultados"])

        processed = min(start + batch_size, len(procesos))
        progress_bar.progress(processed / len(procesos))
        status_text.text(f"Procesados {processed}/{len(procesos)} procesos")

    status_text.text("Evaluacion completada")
    return all_results


health_data = get_health()

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Abodi ML</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Predictor de vencimiento de terminos para procesos judiciales.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if health_data:
        st.success("API conectada")
        if health_data.get("model_f1"):
            st.metric("F1 del modelo", f"{health_data['model_f1']:.4f}")
        st.metric("Predicciones API", health_data.get("total_predictions", 0))
    else:
        st.error("API no disponible")
        st.caption("Levanta el stack Docker o inicia FastAPI localmente.")

    st.divider()
    st.caption("Servicios de la demo")
    st.code(
        "API      http://localhost:8000\n"
        "App      http://localhost:8501\n"
        "MLflow   http://localhost:5000\n"
        "Grafana  http://localhost:3000",
        language="text",
    )


render_header(health_data)

tab_eval, tab_batch, tab_health = st.tabs(
    ["Evaluacion individual", "Carga masiva", "Salud del sistema"]
)

with tab_eval:
    st.markdown('<div class="section-title">Evaluacion individual</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Completa los datos del proceso y evalua el riesgo operativo.</p>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("risk_form"):
            st.markdown("#### Datos del proceso")
            tipo_proceso = st.selectbox("Tipo de proceso", TIPO_PROCESO_OPTIONS)
            tipo_actuacion = st.selectbox(
                "Ultima actuacion",
                TIPO_ACTUACION_OPTIONS,
                format_func=display_label,
            )
            ciudad = st.selectbox("Ciudad", CIUDAD_OPTIONS, format_func=display_label)
            despacho = st.selectbox("Despacho o juzgado", DESPACHO_OPTIONS)
            plan = st.selectbox("Plan de suscripcion", PLAN_OPTIONS, index=1)

            st.markdown("#### Variables operativas")
            c1, c2 = st.columns(2)
            with c1:
                dias = st.number_input("Dias sin actividad", min_value=0, max_value=365, value=15)
                num_partes = st.number_input("Numero de partes", min_value=1, max_value=50, value=2)
            with c2:
                total_acts = st.number_input("Total actuaciones", min_value=0, max_value=200, value=3)
                frecuencia = st.number_input(
                    "Frecuencia act/mes",
                    min_value=0.0,
                    max_value=100.0,
                    value=1.5,
                    step=0.1,
                )
            tiene_termino = st.checkbox("La ultima actuacion tiene termino legal asociado", value=True)

            submitted = st.form_submit_button("Evaluar riesgo", use_container_width=True)

    with result_col:
        if submitted:
            payload = build_payload(
                tipo_proceso,
                tipo_actuacion,
                ciudad,
                despacho,
                int(dias),
                int(num_partes),
                int(total_acts),
                float(frecuencia),
                bool(tiene_termino),
                plan,
            )
            with st.spinner("Consultando modelo..."):
                try:
                    response = requests.post(f"{API_URL}/predict/risk", json=payload, timeout=15)
                    response.raise_for_status()
                    render_risk_result(response.json(), payload)
                except requests.exceptions.ConnectionError:
                    st.error(f"No se puede conectar a la API en {API_URL}.")
                except requests.RequestException as exc:
                    st.error(f"La API retorno un error: {exc}")
        else:
            st.markdown("#### Resultado")
            st.info("Completa el formulario y ejecuta la evaluacion para ver la probabilidad de riesgo.")
            st.markdown(
                """
                El resultado muestra:

                - Probabilidad estimada.
                - Nivel de riesgo.
                - Variables que mas influyen.
                - Recomendacion operativa.
                """
            )

with tab_batch:
    st.markdown('<div class="section-title">Carga masiva</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Evalua varios procesos desde un CSV y exporta la priorizacion.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.85, 1.35], gap="large")

    with left:
        st.markdown("#### Plantilla CSV")
        template = sample_dataframe()
        export_template = template.copy()
        export_template["tipo_ultima_actuacion"] = export_template["tipo_ultima_actuacion"].map(display_label)
        export_template["ciudad"] = export_template["ciudad"].map(display_label)
        csv_example = export_template.to_csv(index=False)
        st.download_button(
            "Descargar plantilla",
            data=csv_example,
            file_name="procesos_ejemplo.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(export_template.head(4), hide_index=True, use_container_width=True)

    with right:
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Archivo cargado: {len(df)} procesos")
                st.dataframe(df.head(12), hide_index=True, use_container_width=True)

                if st.button("Evaluar archivo", type="primary", use_container_width=True):
                    try:
                        results = run_batch_prediction(df)
                        results_df = pd.DataFrame(
                            [
                                {
                                    "Riesgo": "Si" if item["riesgo"] == 1 else "No",
                                    "Probabilidad": item["probabilidad"],
                                    "Nivel": item["nivel"],
                                }
                                for item in results
                            ]
                        )
                        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                        en_riesgo = int((results_df["Riesgo"] == "Si").sum())
                        riesgo_alto = int((results_df["Nivel"] == "Alto").sum())

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Procesos", len(output_df))
                        m2.metric("En riesgo", en_riesgo)
                        m3.metric("Riesgo alto", riesgo_alto)

                        st.dataframe(output_df, hide_index=True, use_container_width=True)
                        csv_result = output_df.to_csv(index=False)
                        st.download_button(
                            "Descargar resultados",
                            data=csv_result,
                            file_name="resultados_riesgo.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except requests.RequestException as exc:
                        st.error(f"No fue posible evaluar el archivo: {exc}")
            except Exception as exc:
                st.error(f"No fue posible leer el CSV: {exc}")

with tab_health:
    st.markdown('<div class="section-title">Salud del sistema</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Indicadores de la demo y estado del modelo publicado.</p>',
        unsafe_allow_html=True,
    )

    latest_health = get_health()
    if latest_health:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API", latest_health["status"])
        c2.metric("Modelo", latest_health.get("model_version") or "N/D")
        c3.metric("F1", f"{latest_health['model_f1']:.4f}" if latest_health.get("model_f1") else "N/D")
        c4.metric("Predicciones", latest_health.get("total_predictions", 0))
    else:
        st.error("La API no esta disponible.")

    st.markdown("#### Limites por plan")
    limits_df = pd.DataFrame(
        [{"Plan": plan, "Limite de procesos": limit} for plan, limit in PLAN_LIMITS.items()]
    )
    st.dataframe(limits_df, hide_index=True, use_container_width=True)

    st.markdown("#### Casos de ejemplo para la demo")
    sample_cases = pd.DataFrame(
        {
            "Radicado": [
                "11001-31-03-001-2024-00123-00",
                "05001-31-03-002-2024-00456-00",
                "76001-31-03-003-2024-00789-00",
                "68001-31-05-004-2024-00218-00",
            ],
            "Tipo": ["Tutela", "Ordinario Civil", "Ejecutivo", "Laboral"],
            "Dias sin actuacion": [25, 120, 8, 18],
            "Estado sugerido": ["Critico", "Atencion", "Al dia", "Atencion"],
        }
    )
    st.dataframe(sample_cases, hide_index=True, use_container_width=True)

    st.markdown("#### Endpoints")
    endpoints_df = pd.DataFrame(
        [
            {"Servicio": "API", "URL": "http://localhost:8000/health"},
            {"Servicio": "MLflow", "URL": "http://localhost:5000"},
            {"Servicio": "Prometheus", "URL": "http://localhost:9090"},
            {"Servicio": "Grafana", "URL": "http://localhost:3000"},
        ]
    )
    st.dataframe(endpoints_df, hide_index=True, use_container_width=True)
