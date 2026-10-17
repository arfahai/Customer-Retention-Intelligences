"""Score every customer eligible at the production cutoff.

Reads : artifacts/churn_pipeline.joblib, artifacts/clv_scores.csv
Writes: artifacts/churn_scores.csv     (features + calibrated probability)
        artifacts/customer_scores.csv  (probability x value = EVaR, priority queue)

Both files come from the same run, so they can never disagree.
"""
from pathlib import Path

import joblib
import pandas as pd

from common import (
    FEATURES,
    PRODUCTION_CUTOFF,
    RETENTION_CAPACITY,
    load_transactions,
    prepare_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_II.xlsx"
ART = ROOT / "artifacts"


def score_population(ct, model, clv):
    s = prepare_snapshot(PRODUCTION_CUTOFF, ct)
    s["CalibratedNoReturnProbability"] = model.predict_proba(s[FEATURES])[:, 1]

    churn = s[["Customer ID", "SnapshotCutoff", *FEATURES, "CalibratedNoReturnProbability"]].copy()

    out = churn[["Customer ID", "SnapshotCutoff", "CalibratedNoReturnProbability"]].merge(
        clv[
            [
                "Customer ID",
                "CLVStatus",
                "HistoricalPurchaseValue",
                "PredictedGrossPurchaseCLV",
                "ValueForRetention",
            ]
        ],
        on="Customer ID",
        how="left",
    )
    out["ExpectedValueAtRisk"] = out.CalibratedNoReturnProbability * out.ValueForRetention

    ids = out.nlargest(RETENTION_CAPACITY, "ExpectedValueAtRisk")["Customer ID"]
    out["RetentionPriority"] = "Standard"
    out.loc[out["Customer ID"].isin(ids), "RetentionPriority"] = "Priority"
    return churn, out


def main():
    raw = load_transactions(DATA)
    ct = raw[raw["Customer ID"].notna()].copy()

    model = joblib.load(ART / "churn_pipeline.joblib")
    clv = pd.read_csv(ART / "clv_scores.csv")

    churn, out = score_population(ct, model, clv)

    assert out["Customer ID"].is_unique
    assert out["ValueForRetention"].notna().all(), "customer without a value proxy"

    churn.to_csv(ART / "churn_scores.csv", index=False)
    out.to_csv(ART / "customer_scores.csv", index=False)
    print("Saved churn_scores.csv and customer_scores.csv:", len(out))


if __name__ == "__main__":
    main()
