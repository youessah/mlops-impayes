from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "telecom_credit.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

HF_REPO = "electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria"

TARGET = "default_risk"
ID_COLS = ["customer_id", "phone_number"]        # identifiants : à exclure
CONSTANT_COLS = ["assessment_date"]              # valeur unique : inutile
LEAKAGE_COLS = ["credit_score", "credit_rating",  # suspects de fuite : à confirmer
                "loan_eligibility", "max_loan_amount_ngn"]