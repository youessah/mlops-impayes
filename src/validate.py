import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.config import RAW_PATH

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(nullable=False),
        "operator": pa.Column(checks=pa.Check.isin(["MTN", "Airtel", "Glo", "9mobile"])),
        "credit_score": pa.Column(checks=pa.Check.in_range(300, 1000)),
        "tenure_months": pa.Column(checks=pa.Check.in_range(0, 120)),
        "avg_monthly_spend_ngn": pa.Column(checks=pa.Check.in_range(0, 100_000)),
        "recharge_frequency_per_month": pa.Column(checks=pa.Check.in_range(0, 31)),
        "payment_consistency_score": pa.Column(checks=pa.Check.in_range(0, 1)),
        "late_payment_count": pa.Column(checks=pa.Check.in_range(0, 30)),
        "account_type": pa.Column(checks=pa.Check.isin(["prepaid", "postpaid"])),
        "default_risk": pa.Column(checks=pa.Check.isin(["low", "medium", "high"])),
    },
    strict=False,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Lève une erreur si les données ne respectent pas le contrat."""
    try:
        RAW_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as err:
        print(err.failure_cases.head(20))
        raise
    print(f"Validation OK : {len(df)} lignes, {df.shape[1]} colonnes")
    print(f"Lignes entièrement dupliquées : {int(df.duplicated().sum())}")
    return df


if __name__ == "__main__":
    validate_raw(pd.read_parquet(RAW_PATH))