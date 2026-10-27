"""Entraînement, suivi MLflow et enregistrement du meilleur modèle."""
import json
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    DVC_FILE,
    EXPERIMENT,
    LEAKAGE_CATEGORICAL,
    LEAKAGE_NUMERIC,
    MLFLOW_URI,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PARAMS_FILE,
    RAW_PATH,
    REGISTERED_MODEL,
    REPORTS_DIR,
    TARGET,
)
from src.features import build_preprocessor, make_target
from src.validate import validate_raw


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_version() -> str:
    """Empreinte DVC (md5) des données utilisées."""
    try:
        with open(DVC_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        return "unknown"


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }
    return metrics, pred


def run_experiment(run_name, model, numeric, categorical, split, seed, md5, tags=None):
    X_train, X_test, y_train, y_test = split
    cols = numeric + categorical
    pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", model)])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags or {})
        mlflow.log_params({k: v for k, v in model.get_params().items() if v is not None})
        mlflow.log_params(
            {
                "features": ",".join(cols),
                "seed": seed,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "data_md5": md5,
            }
        )

        pipe.fit(X_train[cols], y_train)
        metrics, pred = evaluate(pipe, X_test[cols], y_test)
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["non-high", "high"], ax=ax, colorbar=False
        )
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        mlflow.log_dict(
            {"confusion_matrix": confusion_matrix(y_test, pred).tolist()},
            "confusion_matrix.json",
        )

        sample = X_test[cols].head(100)
        try:
            signature = infer_signature(sample, pipe.predict_proba(sample)[:, 1])
        except Exception as exc:
            print(f"(signature non inférée : {exc})")
            signature = None
        info = mlflow.sklearn.log_model(pipe, name="model", signature=signature, serialization_format="cloudpickle")

    print(f"[{run_name}] " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return {"name": run_name, "run_id": run.info.run_id, "model_uri": info.model_uri,
            "metrics": metrics, "pipe": pipe}


def main() -> None:
    plt.switch_backend("Agg")
    params = load_params()
    seed = params["data"]["seed"]
    md5 = data_version()

    df = pd.read_parquet(RAW_PATH)
    validate_raw(df)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Doublons supprimés : {n_before - len(df)}")
    df["loan_eligibility"] = df["loan_eligibility"].astype(int)

    y = make_target(df)
    X = df.drop(columns=[TARGET])
    split = train_test_split(
        X, y, test_size=params["data"]["test_size"], stratify=y, random_state=seed
    )
    y_train = split[2]
    print(f"Part de la classe 'high' : {y.mean():.3f}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    neg_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    lr = LogisticRegression(class_weight="balanced", random_state=seed,
                            **params["models"]["logreg"])
    xgb_params = params["models"]["xgboost"]

    def new_xgb():
        return XGBClassifier(scale_pos_weight=neg_pos, random_state=seed, n_jobs=-1,
                             eval_metric="logloss", **xgb_params)

    candidates = [
        run_experiment("logreg_baseline", lr, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                       split, seed, md5, {"stage": "candidate"}),
        run_experiment("xgboost", new_xgb(), NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                       split, seed, md5, {"stage": "candidate"}),
    ]

    # Démonstration de fuite de données : jamais utilisé en production
    run_experiment(
        "xgboost_WITH_leakage_cols", new_xgb(),
        NUMERIC_FEATURES + LEAKAGE_NUMERIC, CATEGORICAL_FEATURES + LEAKAGE_CATEGORICAL,
        split, seed, md5, {"stage": "leakage_demo", "do_not_deploy": "true"},
    )

    best = max(candidates, key=lambda r: r["metrics"]["f1"])
    min_f1 = params["gate"]["min_f1"]
    print(f"\nMeilleur modèle : {best['name']} (F1 = {best['metrics']['f1']:.3f})")

    if best["metrics"]["f1"] < min_f1:
        print(f"PORTE DE VALIDATION ÉCHOUÉE : F1 < {min_f1}. Modèle non enregistré.")
        sys.exit(1)

    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL, "staging", version.version)
    print(f"Modèle enregistré : {REGISTERED_MODEL} v{version.version} (alias 'staging')")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipe"], MODELS_DIR / "model.joblib")
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": best["name"], "run_id": best["run_id"], "data_md5": md5,
                   "registry_version": version.version, **best["metrics"]}, f, indent=2)
    print("Modèle sauvegardé : models/model.joblib | métriques : reports/metrics.json")


if __name__ == "__main__":
    main()