import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE, ADASYN
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

PROCESSED_PATH = Path("data/processed")
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "Abodi_Vencimiento_Terminos"

X_train = np.load(PROCESSED_PATH / "X_train.npy")
X_test = np.load(PROCESSED_PATH / "X_test.npy")
y_train = np.load(PROCESSED_PATH / "y_train.npy")
y_test = np.load(PROCESSED_PATH / "y_test.npy")
feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Train distribution: {np.bincount(y_train.astype(int))}")
print(f"Test distribution: {np.bincount(y_test.astype(int))}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

experiments = [
    {
        "name": "Baseline - class_weight balanced",
        "params": {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1,
                    "scale_pos_weight": np.sum(y_train == 0) / np.sum(y_train == 1),
                    "subsample": 0.8, "colsample_bytree": 0.8},
        "use_smote": False,
    },
    {
        "name": "SMOTE + XGBoost profundo",
        "params": {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8},
        "use_smote": True,
        "smote_type": "smote",
    },
    {
        "name": "ADASYN + XGBoost regularizado",
        "params": {"max_depth": 3, "n_estimators": 150, "learning_rate": 0.01,
                    "reg_lambda": 2.0, "reg_alpha": 1.0,
                    "subsample": 0.7, "colsample_bytree": 0.7},
        "use_smote": True,
        "smote_type": "adasyn",
    },
]

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sin Riesgo", "Riesgo"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, feature_names, save_path, max_features=15):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:max_features]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Importancia")
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        print(f"\n=== Ejecutando: {exp['name']} ===")

        mlflow.log_params(exp["params"])

        X_train_exp, y_train_exp = X_train.copy(), y_train.copy()

        if exp.get("use_smote"):
            smote_type = exp.get("smote_type", "smote")
            if smote_type == "smote":
                sampler = SMOTE(random_state=42)
            else:
                sampler = ADASYN(random_state=42)
            X_train_exp, y_train_exp = sampler.fit_resample(X_train_exp, y_train_exp)
            mlflow.log_param("sampler", smote_type)
            mlflow.log_param("sampler_k_neighbors", sampler.k_neighbors if hasattr(sampler, "k_neighbors") else 5)
            print(f"  After {smote_type}: {np.bincount(y_train_exp.astype(int))}")

        model = xgb.XGBClassifier(
            **exp["params"],
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )

        model.fit(X_train_exp, y_train_exp)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc_roc": auc,
        })

        cm_path = PROCESSED_PATH / f"cm_{exp['name'].replace(' ', '_').replace('-', '_')}.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path), "confusion_matrix")

        fi_path = PROCESSED_PATH / f"fi_{exp['name'].replace(' ', '_').replace('-', '_')}.png"
        plot_feature_importance(model, feature_cols, fi_path)
        mlflow.log_artifact(str(fi_path), "feature_importance")

        mlflow.sklearn.log_model(model, "model", registered_model_name="Abodi_Risk_Classifier")

        model_path = PROCESSED_PATH / f"model_{exp['name'].replace(' ', '_').replace('-', '_')}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), "models")

print("\n=== Entrenamiento completado ===")
print(f"Ejecuta 'mlflow ui' para ver los resultados.")
