from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "telecom_credit.parquet"
DVC_FILE = ROOT / "data" / "raw" / "telecom_credit.parquet.dvc"
PARAMS_FILE = ROOT / "params.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HF_REPO = "electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria"

TARGET = "default_risk"
POSITIVE_CLASS = "high"          # impayé = risque élevé

NUMERIC_FEATURES = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
]
CATEGORICAL_FEATURES = ["operator", "account_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ID_COLS = ["customer_id", "phone_number"]
CONSTANT_COLS = ["assessment_date"]
LEAKAGE_NUMERIC = ["credit_score", "max_loan_amount_ngn", "loan_eligibility"]
LEAKAGE_CATEGORICAL = ["credit_rating"]
LEAKAGE_COLS = LEAKAGE_NUMERIC + LEAKAGE_CATEGORICAL

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "impayes-default-risk"
REGISTERED_MODEL = "impayes-default-risk"