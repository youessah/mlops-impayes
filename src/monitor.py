"""Monitoring : dérive des données et des scores, performance, latence, décision."""
import argparse
import json
import subprocess

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PARAMS_FILE,
    RAW_PATH,
    REPORTS_DIR,
    ROOT,
)

LOG_PATH = ROOT / "logs" / "predictions.jsonl"
LABEL_PATH = ROOT / "logs" / "labels.jsonl"
EPS = 1e-4


def read_jsonl(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(rows)


def _psi(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    ref = np.clip(ref_counts / ref_counts.sum(), EPS, None)
    cur = np.clip(cur_counts / cur_counts.sum(), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    return _psi(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    ref_counts = np.array([(ref == c).sum() for c in cats], dtype=float)
    cur_counts = np.array([(cur == c).sum() for c in cats], dtype=float)
    return _psi(ref_counts, cur_counts)


def level(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return "CRITICAL"
    if value >= warn:
        return "WARNING"
    return "OK"


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path) -> str | None:
    """Rapport Evidently, généré dans l'environnement isolé .venv-evidently."""
    python = ROOT / ".venv-evidently" / "Scripts" / "python.exe"
    if not python.exists():
        python = ROOT / ".venv-evidently" / "bin" / "python"
    if not python.exists():
        return None
    REPORTS_DIR.mkdir(exist_ok=True)
    ref_csv = REPORTS_DIR / "_ref.csv"
    cur_csv = REPORTS_DIR / "_cur.csv"
    ref.to_csv(ref_csv, index=False)
    cur.to_csv(cur_csv, index=False)
    script = ROOT / "src" / "evidently_report.py"
    proc = subprocess.run(
        [str(python), str(script), str(ref_csv), str(cur_csv), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-600:])
    return proc.stdout.strip() or "Evidently"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-evidently", action="store_true")
    args = parser.parse_args()

    with open(PARAMS_FILE, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["monitoring"]
    with open(REPORTS_DIR / "metrics.json", encoding="utf-8") as fh:
        f1_ref = json.load(fh)["f1"]
    model = joblib.load(MODELS_DIR / "model.joblib")

    logs = read_jsonl(LOG_PATH)
    if len(logs) < cfg["min_samples"]:
        print(f"Pas assez de prédictions ({len(logs)} < {cfg['min_samples']}).")
        return
    logs = logs.tail(cfg["window"]).reset_index(drop=True)
    cur = pd.DataFrame(list(logs["features"]))[FEATURES]

    ref = pd.read_parquet(RAW_PATH, columns=FEATURES).sample(5000, random_state=42)
    ref = ref.reset_index(drop=True)
    ref_scores = model.predict_proba(ref)[:, 1]
    cur_scores = logs["probability"].to_numpy()

    # 1. Qualité des données
    missing_share = float(cur.isna().mean().max())
    quality_ok = missing_share <= cfg["max_missing_share"]

    # 2. Dérive des variables
    drift = {}
    for col in NUMERIC_FEATURES:
        ref_v = ref[col].to_numpy(dtype=float)
        cur_v = cur[col].to_numpy(dtype=float)
        drift[col] = {
            "psi": psi_numeric(ref_v, cur_v),
            "ks": float(stats.ks_2samp(ref_v, cur_v).statistic),
            "ref_mean": float(ref_v.mean()),
            "cur_mean": float(cur_v.mean()),
        }
    for col in CATEGORICAL_FEATURES:
        drift[col] = {"psi": psi_categorical(ref[col], cur[col])}
    for info in drift.values():
        info["level"] = level(info["psi"], cfg["psi_warn"], cfg["psi_crit"])
    n_crit = sum(1 for info in drift.values() if info["level"] == "CRITICAL")

    # 3. Dérive des scores
    score_psi = psi_numeric(ref_scores, cur_scores)
    score_level = level(score_psi, cfg["psi_warn"], cfg["psi_crit"])

    # 4. Latence
    p95 = float(np.percentile(logs["latency_ms"], 95))
    latency_level = level(p95, cfg["latency_p95_warn_ms"], cfg["latency_p95_crit_ms"])

    # 5. Performance (étiquettes différées)
    perf = None
    labels = read_jsonl(LABEL_PATH)
    if not labels.empty:
        merged = logs.merge(labels[["request_id", "label"]], on="request_id")
        if len(merged) >= cfg["min_labels"]:
            y_true, y_pred = merged["label"], merged["prediction"]
            f1 = float(f1_score(y_true, y_pred))
            perf = {
                "n_labels": int(len(merged)),
                "f1": f1,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_drop": float(f1_ref - f1),
            }
            perf["level"] = level(perf["f1_drop"], cfg["f1_drop_warn"], cfg["f1_drop_crit"])

    # 6. Décision
    data_drift_confirmed = n_crit >= cfg["min_drifting_features"] or score_level == "CRITICAL"
    perf_critical = perf is not None and perf["level"] == "CRITICAL"
    if not quality_ok:
        action = "CORRIGER LA SOURCE DE DONNEES (ne pas réentraîner)"
    elif data_drift_confirmed or perf_critical:
        action = "DECLENCHER LE PIPELINE DE REENTRAINEMENT (validation + canari avant promotion)"
    elif latency_level == "CRITICAL":
        action = "DIMENSIONNER / OPTIMISER LE SERVICE (problème technique, pas de réentraînement)"
    else:
        action = "AUCUNE ACTION"
    levels = [score_level, latency_level] + [i["level"] for i in drift.values()]
    if perf:
        levels.append(perf["level"])
    status = "CRITICAL" if "CRITICAL" in levels else ("WARNING" if "WARNING" in levels else "OK")

    result = {
        "status": status,
        "action": action,
        "n_predictions": int(len(logs)),
        "quality": {"max_missing_share": missing_share, "ok": quality_ok},
        "feature_drift": drift,
        "score_drift": {"psi": score_psi, "level": score_level},
        "latency_p95_ms": {"value": p95, "level": latency_level},
        "performance": perf,
        "f1_reference": f1_ref,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "monitoring_report.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"\n=== MONITORING : {status} ({len(logs)} dernières prédictions) ===")
    print(f"{'variable':32s} {'PSI':>7s} {'KS':>6s}  niveau")
    for col, info in drift.items():
        ks = f"{info['ks']:.2f}" if "ks" in info else "-"
        print(f"{col:32s} {info['psi']:7.3f} {ks:>6s}  {info['level']}")
    print(f"{'score du modèle':32s} {score_psi:7.3f} {'-':>6s}  {score_level}")
    print(f"latence p95 : {p95:.1f} ms ({latency_level})")
    if perf:
        print(
            f"F1 = {perf['f1']:.3f} (référence {f1_ref:.3f}, baisse {perf['f1_drop']:+.3f}) "
            f"| précision {perf['precision']:.3f} | rappel {perf['recall']:.3f} "
            f"| {perf['n_labels']} étiquettes ({perf['level']})"
        )
    else:
        print("Performance : pas assez d'étiquettes.")
    print(f"qualité des données : {'OK' if quality_ok else 'PROBLEME'}")
    print(f"ACTION : {action}")

    if not args.no_evidently:
        try:
            used = evidently_report(ref, cur, REPORTS_DIR / "drift_report.html")
            if used:
                print(f"Rapport Evidently : reports/drift_report.html ({used})")
            else:
                print("Environnement .venv-evidently absent : rapport HTML ignoré (la décision reste valide).")
        except Exception as exc:
            print(f"Rapport Evidently non généré : {exc}")


if __name__ == "__main__":
    main()