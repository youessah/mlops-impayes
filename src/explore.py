# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:44:29 2026

@author: pc
"""

import pandas as pd

from src.config import RAW_PATH, TARGET

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

df = pd.read_parquet(RAW_PATH)

print("== Forme ==", df.shape)
print("\n== Types ==\n", df.dtypes)
print("\n== Valeurs manquantes ==\n", df.isna().sum())
print("\n== Doublons customer_id ==", df.duplicated(subset="customer_id").sum())
print("\n== assessment_date : valeurs uniques ==", df["assessment_date"].nunique())

print("\n== Répartition de la cible ==")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))

print("\n== Moyennes par classe de risque ==")
cols = ["credit_score", "tenure_months", "avg_monthly_spend_ngn",
        "recharge_frequency_per_month", "payment_consistency_score",
        "late_payment_count", "max_loan_amount_ngn"]
print(df.groupby(TARGET)[cols].mean().round(2))

print("\n== credit_rating x risque (en %) ==")
print(pd.crosstab(df["credit_rating"], df[TARGET], normalize="index").round(3))

print("\n== loan_eligibility x risque (en %) ==")
print(pd.crosstab(df["loan_eligibility"], df[TARGET], normalize="index").round(3))