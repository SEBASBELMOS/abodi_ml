import mlflow
import joblib
import json
from pathlib import Path

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "Abodi_Vencimiento_Terminos"
PROCESSED_PATH = Path("data/processed")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experimento '{EXPERIMENT_NAME}' no encontrado. Ejecuta train.py primero.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
)

if runs.empty:
    raise ValueError("No se encontraron runs. Ejecuta train.py primero.")

best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_f1 = best_run["metrics.f1_score"]
best_accuracy = best_run["metrics.accuracy"]
best_recall = best_run["metrics.recall"]
best_precision = best_run["metrics.precision"]
best_auc = best_run["metrics.auc_roc"]
best_run_name = best_run.get("tags.mlflow.runName", "unknown")

print("=== Mejor modelo encontrado ===")
print(f"  Run ID: {best_run_id}")
print(f"  Run Name: {best_run_name}")
print(f"  Accuracy:  {best_accuracy:.4f}")
print(f"  Precision: {best_precision:.4f}")
print(f"  Recall:    {best_recall:.4f}")
print(f"  F1-Score:  {best_f1:.4f}")
print(f"  AUC-ROC:   {best_auc:.4f}")

model_name = "Abodi_Risk_Classifier"
model_uri = f"runs:/{best_run_id}/model"

try:
    result = mlflow.register_model(model_uri, model_name)
    print(f"  Versión {result.version} registrada en Model Registry")
except Exception as e:
    print(f"  Registro en MLflow falló (no crítico): {e}")

model = mlflow.sklearn.load_model(model_uri)
model_path = PROCESSED_PATH / "best_model.pkl"
joblib.dump(model, model_path)
print(f"  Modelo guardado localmente en {model_path}")

metadata = {
    "model_name": model_name,
    "version": "Production",
    "run_id": best_run_id,
    "run_name": best_run_name,
    "metrics": {
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1_score": best_f1,
        "auc_roc": best_auc,
    },
}

with open(PROCESSED_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Metadata guardada en {PROCESSED_PATH / 'metadata.json'}")
print("\n=== Selección completada ===")
