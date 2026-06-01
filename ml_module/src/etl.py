import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import joblib

RAW_PATH = Path("data/raw/procesos_judiciales.csv")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

print(f"Registros originales: {len(df)}")
print(f"Target distribution:\n{df['riesgo_vencimiento'].value_counts(normalize=True)}")

cat_features = ["tipo_proceso", "tipo_ultima_actuacion", "ciudad", "despacho", "plan_suscripcion"]
num_features = ["dias_sin_actividad", "num_partes", "total_actuaciones",
                "frecuencia_actualizaciones", "tiene_termino_legal"]

# Feature engineering: pct de plazo consumido (usando el plazo real generado)
df["pct_plazo_consumido"] = np.where(
    df["plazo_dias_calendario"] > 0,
    df["dias_sin_actividad"] / df["plazo_dias_calendario"],
    0
)
df["pct_plazo_consumido"] = df["pct_plazo_consumido"].clip(0, 5)

num_features_extended = num_features + ["pct_plazo_consumido"]

# Label encoding for ordinal-like categoricals (despacho por congestión)
despacho_order = [
    "Juzgado Civil Municipal", "Juzgado Penal Municipal", "Juzgado de Familia",
    "Juzgado Civil del Circuito", "Juzgado Laboral Municipal",
    "Juzgado Laboral del Circuito", "Juzgado Administrativo", "Tribunal Administrativo",
]
df["despacho_encoded"] = df["despacho"].apply(
    lambda x: despacho_order.index(x) if x in despacho_order else len(despacho_order)
)

# One-hot encoding for categoricals
df_encoded = pd.get_dummies(df, columns=["tipo_proceso", "ciudad", "plan_suscripcion",
                                          "tipo_ultima_actuacion"], drop_first=True)

# Drop original columns
drop_cols = ["despacho", "plazo_dias_calendario"]
feature_cols = [c for c in df_encoded.columns if c not in drop_cols + ["riesgo_vencimiento"]]

X = df_encoded[feature_cols].values.astype(np.float64)
y = df_encoded["riesgo_vencimiento"].values

# Scale numerical features
scaler = StandardScaler()
num_indices = [list(df_encoded.columns).index(c) for c in num_features_extended]
# Get the actual column indices in the encoded dataframe
encoded_cols = list(df_encoded.columns)
num_cols_in_encoded = [c for c in num_features_extended if c in encoded_cols]
num_idx = [encoded_cols.index(c) for c in num_cols_in_encoded]

X_num = X[:, num_idx]
X_scaled_num = scaler.fit_transform(X_num)
X[:, num_idx] = X_scaled_num

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save artifacts
joblib.dump(scaler, PROCESSED_PATH / "scaler.pkl")
joblib.dump(feature_cols, PROCESSED_PATH / "feature_cols.pkl")
joblib.dump(despacho_order, PROCESSED_PATH / "despacho_order.pkl")
joblib.dump(num_idx, PROCESSED_PATH / "num_idx.pkl")

np.save(PROCESSED_PATH / "X_train.npy", X_train)
np.save(PROCESSED_PATH / "X_test.npy", X_test)
np.save(PROCESSED_PATH / "y_train.npy", y_train)
np.save(PROCESSED_PATH / "y_test.npy", y_test)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")
print(f"y_test distribution: {np.bincount(y_test)}")
print(f"\nTotal features: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print("ETL completado. Archivos guardados en data/processed/")
