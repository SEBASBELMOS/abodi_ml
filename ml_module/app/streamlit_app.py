import streamlit as st
import requests
import pandas as pd
import json
import io
import time

st.set_page_config(
    page_title="Abodi ML - Vencimiento de Términos",
    page_icon="⚖️",
    layout="wide",
)

API_URL = "http://localhost:8000"

st.sidebar.title("⚖️ Abodi ML")
st.sidebar.markdown("### Predictor de Vencimiento de Términos")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo:** XGBoost Clasificador")
st.sidebar.markdown("**Versión API:** v1.0.0")

tab1, tab2, tab3 = st.tabs(["🔍 Evaluación Individual", "📂 Carga Masiva", "📊 Dashboard de Salud"])

TIPO_PROCESO_OPTIONS = [
    "Tutela", "Ordinario Civil", "Ejecutivo", "Laboral", "Administrativo"
]
TIPO_ACTUACION_OPTIONS = [
    "Auto admisorio demanda", "Traslado excepciones", "Traslado recurso",
    "Sentencia primera instancia", "Notificación por estado", "Auto de pruebas",
    "Fijación audiencia", "Providencia interlocutoria", "Constancia secretarial",
    "Oficio comisorio",
]
CIUDAD_OPTIONS = [
    "Bogotá", "Medellín", "Cali", "Barranquilla", "Manizales", "Bucaramanga", "Otras"
]
DESPACHO_OPTIONS = [
    "Juzgado Civil Municipal", "Juzgado Penal Municipal", "Juzgado de Familia",
    "Juzgado Civil del Circuito", "Juzgado Laboral Municipal",
    "Juzgado Laboral del Circuito", "Juzgado Administrativo", "Tribunal Administrativo",
]
PLAN_OPTIONS = ["Freemium", "Pro", "Premium", "Firma"]

PLAN_LIMITS = {
    "Freemium": 5,
    "Pro": 50,
    "Premium": 200,
    "Firma": 999999,
}

with tab1:
    st.header("Evaluación Individual de Riesgo")
    st.markdown("Ingresa los datos del proceso judicial para determinar el riesgo de vencimiento de términos.")

    col1, col2 = st.columns(2)

    with col1:
        tipo_proceso = st.selectbox("Tipo de Proceso", TIPO_PROCESO_OPTIONS, key="eval_tipo")
        tipo_actuacion = st.selectbox("Última Actuación", TIPO_ACTUACION_OPTIONS, key="eval_act")
        ciudad = st.selectbox("Ciudad", CIUDAD_OPTIONS, key="eval_ciudad")
        despacho = st.selectbox("Despacho / Juzgado", DESPACHO_OPTIONS, key="eval_despacho")
        plan = st.selectbox("Plan de Suscripción", PLAN_OPTIONS, key="eval_plan")

    with col2:
        dias = st.number_input("Días sin Actividad", min_value=0, max_value=365, value=10, key="eval_dias")
        num_partes = st.number_input("Número de Partes", min_value=1, max_value=50, value=2, key="eval_partes")
        total_acts = st.number_input("Total Actuaciones", min_value=0, max_value=200, value=5, key="eval_acts")
        frecuencia = st.number_input("Frecuencia (act/mes)", min_value=0.0, max_value=100.0, value=1.5, step=0.1, key="eval_frec")
        tiene_termino = st.checkbox("Tiene Término Legal Asociado", value=True, key="eval_termino")

    if st.button("🔍 Evaluar Riesgo", type="primary", use_container_width=True):
        payload = {
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

        with st.spinner("Evaluando riesgo..."):
            try:
                resp = requests.post(f"{API_URL}/predict/risk", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    prob = result["probabilidad"]
                    nivel = result["nivel"]
                    if nivel == "Alto":
                        color = "#ff4b4b"
                        emoji = "🔴"
                    elif nivel == "Medio":
                        color = "#ffa500"
                        emoji = "🟡"
                    else:
                        color = "#00cc66"
                        emoji = "🟢"

                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:20px; border-radius:10px; border:2px solid {color};">
                            <h1>{emoji}</h1>
                            <h2 style="color:{color};">{nivel}</h2>
                            <h3>Riesgo de Vencimiento</h3>
                            <p style="font-size:36px; font-weight:bold; color:{color};">{prob:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_b:
                    st.markdown("### 📋 Datos del Caso")
                    st.markdown(f"- **Tipo:** {tipo_proceso}")
                    st.markdown(f"- **Actuación:** {tipo_actuacion}")
                    st.markdown(f"- **Ciudad:** {ciudad}")
                    st.markdown(f"- **Despacho:** {despacho}")
                    st.markdown(f"- **Plan:** {plan}")
                    st.markdown(f"- **Días sin actividad:** {dias}")

                with col_c:
                    st.markdown("### 📊 Factores de Riesgo")
                    factores = result.get("factores_riesgo", [])
                    if factores:
                        for f in factores[:5]:
                            direction_text = "⬆️" if f["direction"] == "aumenta_riesgo" else "⬇️"
                            st.markdown(f"- {direction_text} **{f['feature']}**")
                    else:
                        st.markdown("*No hay factores detallados disponibles*")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ No se puede conectar a la API en {API_URL}. ¿Está corriendo?")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab2:
    st.header("Carga Masiva de Procesos")
    st.markdown("Sube un archivo CSV con múltiples procesos para evaluar el riesgo simultáneamente.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Formato del CSV")
        st.markdown("""
        El archivo debe contener las siguientes columnas:
        - `tipo_proceso`
        - `tipo_ultima_actuacion`
        - `ciudad`
        - `despacho`
        - `dias_sin_actividad`
        - `num_partes`
        - `total_actuaciones`
        - `frecuencia_actualizaciones`
        - `tiene_termino_legal`
        - `plan_suscripcion`
        """)

        example_df = pd.DataFrame({
            "tipo_proceso": ["Tutela", "Ordinario Civil"],
            "tipo_ultima_actuacion": ["Notificación por estado", "Auto admisorio demanda"],
            "ciudad": ["Bogotá", "Medellín"],
            "despacho": ["Juzgado Civil Municipal", "Juzgado Civil del Circuito"],
            "dias_sin_actividad": [15, 45],
            "num_partes": [2, 3],
            "total_actuaciones": [3, 12],
            "frecuencia_actualizaciones": [1.5, 2.0],
            "tiene_termino_legal": [1, 1],
            "plan_suscripcion": ["Pro", "Premium"],
        })
        csv_example = example_df.to_csv(index=False)
        st.download_button("📥 Descargar CSV de ejemplo", data=csv_example, file_name="procesos_ejemplo.csv", mime="text/csv")

    with col2:
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df)} procesos")

                st.dataframe(df.head(10), use_container_width=True)

                if st.button("🚀 Evaluar Todos", type="primary", use_container_width=True):
                    procesos = df.to_dict(orient="records")
                    payload = {"procesos": procesos}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    batch_size = 50
                    all_results = []
                    total = len(procesos)

                    for i in range(0, total, batch_size):
                        batch = procesos[i:i+batch_size]
                        try:
                            resp = requests.post(f"{API_URL}/predict/batch", json={"procesos": batch}, timeout=30)
                            resp.raise_for_status()
                            result = resp.json()
                            all_results.extend(result["resultados"])
                        except Exception as e:
                            st.error(f"Error en lote {i//batch_size + 1}: {e}")

                        progress = min((i + batch_size) / total, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Procesados {min(i + batch_size, total)}/{total}")

                    progress_bar.progress(1.0)
                    status_text.text("✅ Evaluación completada")

                    if all_results:
                        results_df = pd.DataFrame([{
                            "Riesgo": "🔴 Sí" if r["riesgo"] == 1 else "🟢 No",
                            "Probabilidad": f"{r['probabilidad']:.1%}",
                            "Nivel": r["nivel"],
                        } for r in all_results])

                        col_a, col_b, col_c = st.columns(3)
                        en_riesgo = sum(1 for r in all_results if r["riesgo"] == 1)
                        col_a.metric("Total Procesos", total)
                        col_b.metric("En Riesgo", en_riesgo, delta=f"{en_riesgo/total:.1%}" if total > 0 else "0%")
                        col_c.metric("Sin Riesgo", total - en_riesgo)

                        st.dataframe(pd.concat([df.reset_index(drop=True), results_df], axis=1), use_container_width=True)

                        csv_result = pd.concat([df, results_df], axis=1).to_csv(index=False)
                        st.download_button("📥 Descargar Resultados", data=csv_result, file_name="resultados_riesgo.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

with tab3:
    st.header("📊 Dashboard de Salud de Procesos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Procesos Monitoreados", "--", "cargando...")
    col2.metric("En Riesgo Alto", "--", "cargando...")
    col3.metric("Críticos", "--", "cargando...")
    col4.metric("Al Día", "--", "cargando...")

    st.markdown("### Estado del Sistema")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            h = health.json()
            st.success(f"✅ API: **{h['status']}**")
            if h.get("model_version"):
                st.info(f"📦 Modelo: {h['model_version']} | F1: {h['model_f1']:.4f}" if h.get("model_f1") else f"📦 Modelo: {h['model_version']}")
            st.info(f"📊 Predicciones totales: {h['total_predictions']}")
    except Exception:
        st.warning("⚠️ No se puede conectar a la API. Inicia `api/router.py` para ver datos en vivo.")

    st.markdown("### Límites por Plan de Suscripción")

    limits_df = pd.DataFrame([
        {"Plan": plan, "Límite Procesos": limit}
        for plan, limit in PLAN_LIMITS.items()
    ])
    st.table(limits_df)

    st.markdown("### Procesos de Ejemplo")
    sample_cases = pd.DataFrame({
        "Radicado": ["11001-31-03-001-2024-00123-00", "05001-31-03-002-2024-00456-00",
                     "76001-31-03-003-2024-00789-00"],
        "Tipo": ["Tutela", "Ordinario Civil", "Ejecutivo"],
        "Días sin Actuación": [25, 120, 8],
        "Estado": ["🔴 Crítico", "🟡 Atención", "🟢 Al día"],
    })
    st.dataframe(sample_cases, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Ayuda")

with st.sidebar.expander("¿Cómo interpretar los resultados?"):
    st.markdown("""
    - **🔴 Alto (>70%):** Riesgo inminente de vencimiento. Actuar de inmediato.
    - **🟡 Medio (50-70%):** Probabilidad considerable. Revisar el caso.
    - **🟢 Bajo (<50%):** Sin riesgo significativo. Monitoreo normal.

    **Factores SHAP:** Muestran qué variables influyeron más en la predicción.
    """)
