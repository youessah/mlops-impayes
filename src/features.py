import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import POSITIVE_CLASS, TARGET


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 = risque élevé (impayé), 0 = sinon."""
    return (df[TARGET] == POSITIVE_CLASS).astype(int)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )