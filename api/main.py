"""API de scoring du risque d'impayé (FastAPI)."""
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from api.schemas import FEATURES, CustomerFeatures, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
_LOG_LOCK = threading.Lock()


def _settings() -> dict:
    return {
        "model_path": Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib")),
        "metrics_path": Path(os.getenv("METRICS_PATH", ROOT / "reports" / "metrics.json")),
        "log_path": Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl")),
        "threshold": float(os.getenv("THRESHOLD", "0.5")),
    }


def _load_meta(path: Path) -> dict:
    unknown = {
        "model_name": "unknown",
        "model_version": "unknown",
        "mlflow_run_id": "unknown",
        "data_md5": "unknown",
        "f1_test": None,
    }
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return unknown
    return {
        "model_name": m.get("model", "unknown"),
        "model_version": f"v{m['registry_version']}" if "registry_version" in m else "unknown",
        "mlflow_run_id": m.get("run_id", "unknown"),
        "data_md5": m.get("data_md5", "unknown"),
        "f1_test": m.get("f1"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = _settings()
    app.state.settings = s
    app.state.model = joblib.load(s["model_path"])   # échec = le service ne démarre pas
    app.state.meta = _load_meta(s["metrics_path"])
    s["log_path"].parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="API de prédiction des impayés",
    version="1.0.0",
    description="Score le risque d'impayé d'un client télécom (default_risk = high).",
    lifespan=lifespan,
)


def _log_prediction(path: Path, record: dict) -> None:
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/health")
def health(request: Request) -> dict:
    st = request.app.state
    return {
        "status": "ok",
        "model_loaded": st.model is not None,
        "model_version": st.meta["model_version"],
    }


@app.get("/model-info")
def model_info(request: Request) -> dict:
    st = request.app.state
    return {**st.meta, "threshold": st.settings["threshold"], "features": FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures, request: Request) -> PredictionResponse:
    start = time.perf_counter()
    st = request.app.state

    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    proba = float(st.model.predict_proba(row)[0, 1])
    threshold = st.settings["threshold"]
    is_high = proba >= threshold

    request_id = uuid.uuid4().hex
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    _log_prediction(
        st.settings["log_path"],
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": st.meta["model_version"],
            "features": payload.model_dump(),
            "probability": round(proba, 6),
            "prediction": int(is_high),
            "latency_ms": latency_ms,
        },
    )

    return PredictionResponse(
        request_id=request_id,
        default_probability=round(proba, 6),
        is_high_risk=is_high,
        threshold=threshold,
        model_version=st.meta["model_version"],
        latency_ms=latency_ms,
    )