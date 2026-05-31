import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathlib import Path

N_SAMPLES = 10_000
SEED = 42
rng = default_rng(SEED)

TIPOS_PROCESO = {
    "Tutela":            {"plazo_min": 3,  "plazo_max": 10,  "peso": 0.30},
    "Ordinario Civil":   {"plazo_min": 10, "plazo_max": 30,  "peso": 0.20},
    "Ejecutivo":         {"plazo_min": 5,  "plazo_max": 15,  "peso": 0.20},
    "Laboral":           {"plazo_min": 10, "plazo_max": 20,  "peso": 0.15},
    "Administrativo":    {"plazo_min": 10, "plazo_max": 30,  "peso": 0.15},
}

ACTUACIONES_CON_TERMINO = {
    "Auto admisorio demanda":   {"termino": True,  "plazo_legal": 10},
    "Traslado excepciones":     {"termino": True,  "plazo_legal": 10},
    "Traslado recurso":         {"termino": True,  "plazo_legal": 5},
    "Sentencia primera instancia": {"termino": True, "plazo_legal": 10},
    "Notificación por estado":  {"termino": True,  "plazo_legal": 3},
    "Auto de pruebas":          {"termino": True,  "plazo_legal": 10},
    "Fijación audiencia":       {"termino": True,  "plazo_legal": 5},
    "Providencia interlocutoria": {"termino": False, "plazo_legal": 0},
    "Constancia secretarial":   {"termino": False, "plazo_legal": 0},
    "Oficio comisorio":         {"termino": False, "plazo_legal": 0},
}

CIUDADES = {
    "Bogotá":       {"peso": 0.35, "congestion": 1.4},
    "Medellín":     {"peso": 0.20, "congestion": 1.1},
    "Cali":         {"peso": 0.15, "congestion": 1.0},
    "Barranquilla": {"peso": 0.10, "congestion": 0.85},
    "Manizales":    {"peso": 0.05, "congestion": 0.6},
    "Bucaramanga":  {"peso": 0.10, "congestion": 0.8},
    "Otras":        {"peso": 0.05, "congestion": 0.5},
}

DESPACHOS_POR_TIPO = {
    "Tutela":            ["Juzgado Civil Municipal", "Juzgado Penal Municipal", "Juzgado de Familia"],
    "Ordinario Civil":   ["Juzgado Civil del Circuito", "Juzgado Civil Municipal"],
    "Ejecutivo":         ["Juzgado Civil del Circuito", "Juzgado Civil Municipal"],
    "Laboral":           ["Juzgado Laboral del Circuito", "Juzgado Laboral Municipal"],
    "Administrativo":    ["Tribunal Administrativo", "Juzgado Administrativo"],
}

PLANES = ["Freemium", "Pro", "Premium", "Firma"]
PESOS_PLANES = [0.30, 0.35, 0.25, 0.10]

LIMITES_PROCESOS = {
    "Freemium": 5,
    "Pro": 50,
    "Premium": 200,
    "Firma": 1_000_000,
}

def to_calendar_days(habiles):
    return int(round(habiles * rng.uniform(1.35, 1.45)))

data = []

for _ in range(N_SAMPLES):
    ciudad = rng.choice(list(CIUDADES.keys()), p=[v["peso"] for v in CIUDADES.values()])
    congestion = CIUDADES[ciudad]["congestion"]

    tipo_proceso = rng.choice(list(TIPOS_PROCESO.keys()), p=[v["peso"] for v in TIPOS_PROCESO.values()])
    plazo_info = TIPOS_PROCESO[tipo_proceso]
    despacho = rng.choice(DESPACHOS_POR_TIPO[tipo_proceso])

    tipo_actuacion = rng.choice(list(ACTUACIONES_CON_TERMINO.keys()))
    actuacion_info = ACTUACIONES_CON_TERMINO[tipo_actuacion]
    tiene_termino = actuacion_info["termino"]
    plazo_actuacion = actuacion_info["plazo_legal"]

    if tiene_termino:
        plazo_dias_habiles = int(rng.integers(max(plazo_actuacion - 2, 1), plazo_actuacion + 3))
        plazo_dias_calendario = to_calendar_days(plazo_dias_habiles)
    else:
        plazo_dias_calendario = 0

    base_dias = rng.exponential(
        scale=15 * congestion * (1.3 if "Ordinario" in tipo_proceso or "Administrativo" in tipo_proceso else 0.7)
    )
    dias_sin_actividad = min(int(base_dias), 365)

    num_partes = int(rng.poisson(2)) + 1
    total_actuaciones = int(rng.gamma(shape=3, scale=5)) + 1
    frecuencia = round(total_actuaciones / max(dias_sin_actividad / 30, 1), 2)
    plan = rng.choice(PLANES, p=PESOS_PLANES)

    if tiene_termino and dias_sin_actividad > plazo_dias_calendario * 0.7:
        riesgo = 1
    else:
        riesgo = 0

    if rng.random() < 0.08:
        riesgo = 1 - riesgo

    data.append({
        "tipo_proceso": tipo_proceso,
        "tipo_ultima_actuacion": tipo_actuacion,
        "ciudad": ciudad,
        "despacho": despacho,
        "dias_sin_actividad": dias_sin_actividad,
        "num_partes": num_partes,
        "total_actuaciones": total_actuaciones,
        "frecuencia_actualizaciones": frecuencia,
        "tiene_termino_legal": 1 if tiene_termino else 0,
        "plan_suscripcion": plan,
        "riesgo_vencimiento": riesgo,
    })

df = pd.DataFrame(data)

print("Distribución target:")
print(df["riesgo_vencimiento"].value_counts(normalize=True))
print(f"\nTotal registros: {len(df)}")

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "procesos_judiciales.csv", index=False)
print(f"Dataset guardado en {output_dir / 'procesos_judiciales.csv'}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nPrimeras filas:\n{df.head()}")
