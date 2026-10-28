"""Simule du trafic vers l'API (phase normale ou phase avec dérive).

Les étiquettes (paiements réels) sont écrites à part, comme elles arriveraient en différé.
"""
import argparse
import json

import httpx
import numpy as np
import pandas as pd

from src.config import FEATURES, NUMERIC_FEATURES, POSITIVE_CLASS, RAW_PATH, ROOT, TARGET

LABEL_PATH = ROOT / "logs" / "labels.jsonl"
RANGES = {
    "tenure_months": (0, 120),
    "avg_monthly_spend_ngn": (0, 100_000),
    "recharge_frequency_per_month": (0, 31),
    "payment_consistency_score": (0, 1),
    "late_payment_count": (0, 30),
}


def apply_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Dérive simulée : dépenses en forte baisse, plus de retards, moins de recharges."""
    out = df.copy()
    out["avg_monthly_spend_ngn"] = out["avg_monthly_spend_ngn"] * rng.uniform(0.35, 0.6, len(out))
    out["late_payment_count"] = out["late_payment_count"] + 2
    out["recharge_frequency_per_month"] = out["recharge_frequency_per_month"] - 4
    for col, (low, high) in RANGES.items():
        out[col] = out[col].clip(low, high)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(RAW_PATH, columns=FEATURES + [TARGET])
    df = df.sample(args.n, random_state=args.seed).reset_index(drop=True)
    labels = (df[TARGET] == POSITIVE_CLASS).astype(int).tolist()
    feats = df[FEATURES]
    if args.drift:
        feats = apply_drift(feats, rng)
    phase = "drift" if args.drift else "normal"

    LABEL_PATH.parent.mkdir(exist_ok=True)
    sent = 0
    with httpx.Client(timeout=10) as client, open(LABEL_PATH, "a", encoding="utf-8") as fh:
        for i, row in feats.iterrows():
            payload = {c: (float(row[c]) if c in NUMERIC_FEATURES else str(row[c])) for c in FEATURES}
            response = client.post(f"{args.url}/predict", json=payload)
            response.raise_for_status()
            request_id = response.json()["request_id"]
            fh.write(json.dumps({"request_id": request_id, "label": labels[i], "phase": phase}) + "\n")
            sent += 1
    print(f"{sent} requêtes envoyées (phase = {phase}) | étiquettes : {LABEL_PATH}")


if __name__ == "__main__":
    main()