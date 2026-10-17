"""Fit BG/NBD + Gamma-Gamma and write customer-level CLV.

Transaction grain: one customer-date transaction (valid purchases summed per day).
Horizon: 12 months, 1% monthly discount, freq="D".
PredictedGrossPurchaseCLV is gross purchase value, not profit.

Output: artifacts/clv_scores.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from common import PRODUCTION_CUTOFF, load_transactions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_II.xlsx"
ART = ROOT / "artifacts"

CLV_MONTHS = 12
MONTHLY_DISCOUNT = 0.01


def customer_date_transactions(ct, cutoff):
    p = ct[(ct.InvoiceDate <= cutoff) & (ct.Transaction_Type == "Valid Purchase")].copy()
    p["TransactionValue"] = p.Quantity * p.Price
    return (
        p.assign(PurchaseDate=p.InvoiceDate.dt.floor("D"))
        .groupby(["Customer ID", "PurchaseDate"], as_index=False)
        .TransactionValue.sum()
    )


def fit_clv_models(ct, cutoff=PRODUCTION_CUTOFF):
    """Return (transactions, rfm summary, BG/NBD model, Gamma-Gamma model, repeat customers)."""
    tx = customer_date_transactions(ct, cutoff)

    s = summary_data_from_transaction_data(
        tx,
        customer_id_col="Customer ID",
        datetime_col="PurchaseDate",
        monetary_value_col="TransactionValue",
        observation_period_end=cutoff,
    ).reset_index()

    bg = BetaGeoFitter(penalizer_coef=0.1)
    bg.fit(s["frequency"], s["recency"], s["T"])

    rep = s[s["frequency"] > 0].copy()
    gg = GammaGammaFitter(penalizer_coef=0.001)
    gg.fit(rep["frequency"], rep["monetary_value"])
    return tx, s, bg, gg, rep


def main():
    raw = load_transactions(DATA)
    ct = raw[raw["Customer ID"].notna()].copy()

    tx, s, bg, gg, rep = fit_clv_models(ct)

    rep["PredictedGrossPurchaseCLV"] = gg.customer_lifetime_value(
        transaction_prediction_model=bg,
        frequency=rep["frequency"],
        recency=rep["recency"],
        T=rep["T"],
        monetary_value=rep["monetary_value"],
        time=CLV_MONTHS,
        discount_rate=MONTHLY_DISCOUNT,
        freq="D",
    )

    hist = (
        tx.groupby("Customer ID")["TransactionValue"]
        .sum()
        .rename("HistoricalPurchaseValue")
        .reset_index()
    )

    out = s.merge(hist, on="Customer ID", how="left").merge(
        rep[["Customer ID", "PredictedGrossPurchaseCLV"]], on="Customer ID", how="left"
    )

    out["CLVStatus"] = np.where(
        out["frequency"] > 0, "Predicted CLV Eligible", "Not Eligible - One-Time Customer"
    )
    out["ValueForRetention"] = np.where(
        out["CLVStatus"] == "Predicted CLV Eligible",
        out["PredictedGrossPurchaseCLV"],
        out["HistoricalPurchaseValue"],
    )

    ART.mkdir(exist_ok=True)
    out.to_csv(ART / "clv_scores.csv", index=False)
    print("Saved clv_scores.csv")


if __name__ == "__main__":
    main()
