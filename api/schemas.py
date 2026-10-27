from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NUMERIC_FEATURES = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
]
CATEGORICAL_FEATURES = ["operator", "account_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class CustomerFeatures(BaseModel):
    """Caractéristiques d'un client (aucun identifiant accepté)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "tenure_months": 12,
                "avg_monthly_spend_ngn": 5000,
                "recharge_frequency_per_month": 4,
                "payment_consistency_score": 0.2,
                "late_payment_count": 6,
                "operator": "MTN",
                "account_type": "prepaid",
            }
        },
    )

    tenure_months: float = Field(..., ge=0, le=120, description="Ancienneté (mois)")
    avg_monthly_spend_ngn: float = Field(..., ge=0, le=100_000, description="Dépense mensuelle moyenne (NGN)")
    recharge_frequency_per_month: float = Field(..., ge=0, le=31, description="Recharges par mois")
    payment_consistency_score: float = Field(..., ge=0, le=1, description="Régularité des paiements (0 à 1)")
    late_payment_count: float = Field(..., ge=0, le=30, description="Nombre de retards de paiement")
    operator: Literal["MTN", "Airtel", "Glo", "9mobile"]
    account_type: Literal["prepaid", "postpaid"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    default_probability: float = Field(..., description="Probabilité de risque élevé (0 à 1)")
    is_high_risk: bool = Field(..., description="True si la probabilité dépasse le seuil")
    threshold: float
    model_version: str
    latency_ms: float