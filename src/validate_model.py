"""Final temporal-test + CLV-holdout validation of the SAVED production artifacts.

Churn : saved model scored on the untouched 9 Jun 2011 snapshot (labels = 10 Jun-6 Dec 2011).
CLV   : BG/NBD + Gamma-Gamma fitted to 9 Jun 2011, compared with realised purchases
        in the following 180 days (repeat customers only).

Writes a versioned evidence package to reports/:
    final_validation_metrics.md / .json / .csv
    calibration_final_test.csv, clv_decile_capture.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from clv import fit_clv_models
from common import (
    FEATURES,
    OUTCOME_DAYS,
    PRODUCTION_CUTOFF,
    RETENTION_CAPACITY,
    TARGET,
    load_transactions,
    prepare_snapshot,
)
from train_churn import development_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_II.xlsx"
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42
N_BOOT = 1000


def churn_metrics(y, p):
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
    }


def bootstrap_ci(y, p, n=N_BOOT):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(churn_metrics(y[idx], p[idx]))
    b = pd.DataFrame(rows)
    return {k: (float(b[k].quantile(0.025)), float(b[k].quantile(0.975))) for k in b.columns}


def churn_validation(ct):
    model = joblib.load(ART / "churn_pipeline.joblib")
    test = prepare_snapshot(PRODUCTION_CUTOFF, ct)
    y = test[TARGET].to_numpy()
    p = model.predict_proba(test[FEATURES])[:, 1]

    model_m = churn_metrics(y, p)
    ci = bootstrap_ci(y, p)

    # Baselines fitted on the same development data as the production model.
    dev = development_frame(ct)
    prevalence = float(dev[TARGET].mean())
    prev_p = np.full(len(y), prevalence)
    recency = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(
        dev[["RecencyDays"]], dev[TARGET]
    )
    rec_p = recency.predict_proba(test[["RecencyDays"]])[:, 1]
    baselines = {
        "Production model": model_m,
        "Recency-only logistic": churn_metrics(y, rec_p),
        "Constant (dev prevalence)": {
            "ROC-AUC": 0.5,
            "PR-AUC": float(y.mean()),
            "Brier": brier_score_loss(y, prev_p),
            "LogLoss": log_loss(y, prev_p),
        },
    }

    # Capacity view: precision / lift in the top-100 by probability.
    order = np.argsort(-p)[:RETENTION_CAPACITY]
    capacity = {
        "capacity": RETENTION_CAPACITY,
        "precision_at_capacity": float(y[order].mean()),
        "base_rate": float(y.mean()),
        "lift_at_capacity": float(y[order].mean() / y.mean()),
    }

    groups = []
    for name, value in [("Single-purchase", 1), ("Established", 0)]:
        m = test["SinglePurchaseHistory"].eq(value).to_numpy()
        groups.append(
            {
                "Group": name,
                "Customers": int(m.sum()),
                "NoReturnRate": float(y[m].mean()),
                "ROC-AUC": float(roc_auc_score(y[m], p[m])),
                "PR-AUC": float(average_precision_score(y[m], p[m])),
            }
        )

    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    cal = (
        pd.DataFrame({"bin": bins, "p": p, "y": y})
        .groupby("bin", observed=True)
        .agg(Customers=("y", "size"), MeanPredicted=("p", "mean"), ObservedRate=("y", "mean"))
        .reset_index()
    )
    cal["bin"] = cal["bin"].astype(str)

    return {
        "cohort": {
            "snapshot_cutoff": str(PRODUCTION_CUTOFF.date()),
            "customers": int(len(test)),
            "no_return_rate": float(y.mean()),
            "mean_predicted_probability": float(p.mean()),
        },
        "metrics": model_m,
        "bootstrap_95ci": ci,
        "baselines": baselines,
        "capacity": capacity,
        "history_groups": groups,
    }, cal


def clv_validation(ct):
    _, _, bg, gg, rep = fit_clv_models(ct, PRODUCTION_CUTOFF)
    end = PRODUCTION_CUTOFF + pd.Timedelta(days=OUTCOME_DAYS)
    h = ct[
        (ct.InvoiceDate > PRODUCTION_CUTOFF)
        & (ct.InvoiceDate <= end)
        & (ct.Transaction_Type == "Valid Purchase")
    ].copy()
    h["TransactionValue"] = h.Quantity * h.Price
    daily = (
        h.assign(PurchaseDate=h.InvoiceDate.dt.floor("D"))
        .groupby(["Customer ID", "PurchaseDate"], as_index=False)
        .TransactionValue.sum()
    )
    actual = (
        daily.groupby("Customer ID")
        .agg(ActualPurchaseDateCount=("PurchaseDate", "nunique"), ActualPurchaseValue=("TransactionValue", "sum"))
        .reset_index()
    )

    b = rep[["Customer ID", "frequency", "recency", "T", "monetary_value"]].copy()
    b["PredCount"] = bg.predict(OUTCOME_DAYS, b["frequency"], b["recency"], b["T"])
    b["PredAvg"] = gg.conditional_expected_average_profit(b["frequency"], b["monetary_value"])
    b["PredValue"] = b["PredCount"] * b["PredAvg"]
    b = b.merge(actual, on="Customer ID", how="left").fillna(
        {"ActualPurchaseDateCount": 0, "ActualPurchaseValue": 0}
    )
    hist = (
        ct[(ct.InvoiceDate <= PRODUCTION_CUTOFF) & (ct.Transaction_Type == "Valid Purchase")]
        .assign(v=lambda d: d.Quantity * d.Price)
        .groupby("Customer ID")["v"]
        .sum()
    )
    b["HistValue"] = b["Customer ID"].map(hist)

    def rmse(a, p):
        return float(mean_squared_error(a, p) ** 0.5)

    summary = [
        {
            "Metric": "Purchase-date count (180d)",
            "PredictedMean": float(b.PredCount.mean()),
            "ActualMean": float(b.ActualPurchaseDateCount.mean()),
            "MAE": float(mean_absolute_error(b.ActualPurchaseDateCount, b.PredCount)),
            "RMSE": rmse(b.ActualPurchaseDateCount, b.PredCount),
        },
        {
            "Metric": "Purchase value (180d, GBP)",
            "PredictedMean": float(b.PredValue.mean()),
            "ActualMean": float(b.ActualPurchaseValue.mean()),
            "MAE": float(mean_absolute_error(b.ActualPurchaseValue, b.PredValue)),
            "RMSE": rmse(b.ActualPurchaseValue, b.PredValue),
        },
    ]

    def decile_table(score):
        d = pd.DataFrame({"s": score, "v": b.ActualPurchaseValue})
        d["Decile"] = pd.qcut(d.s.rank(method="first"), 10, labels=False) + 1
        t = d.groupby("Decile").agg(Customers=("v", "size"), RealizedValue=("v", "sum"))
        t = t.sort_index(ascending=False)
        t["ShareOfRealizedValue"] = t.RealizedValue / t.RealizedValue.sum()
        return t.reset_index()

    deciles = decile_table(b.PredValue)
    base_deciles = decile_table(b.HistValue)
    out = {
        "customers_repeat": int(len(b)),
        "summary": summary,
        "top_decile_share_predicted_value": float(deciles.ShareOfRealizedValue.iloc[0]),
        "top_decile_share_historical_value": float(base_deciles.ShareOfRealizedValue.iloc[0]),
        "spearman_predicted_vs_actual_value": float(b.PredValue.corr(b.ActualPurchaseValue, method="spearman")),
    }
    return out, deciles


def pct(x):
    return f"{100 * x:.1f}%"


def md_table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_reports(churn, cal, clv, deciles):
    REPORTS.mkdir(exist_ok=True)
    cal.to_csv(REPORTS / "calibration_final_test.csv", index=False)
    deciles.to_csv(REPORTS / "clv_decile_capture.csv", index=False)
    (REPORTS / "final_validation_metrics.json").write_text(
        json.dumps({"churn": churn, "clv": clv}, indent=2) + "\n"
    )

    rows = []
    for k, v in churn["metrics"].items():
        lo, hi = churn["bootstrap_95ci"][k]
        rows.append(("churn_final_test", k, v, lo, hi))
    for name, m in churn["baselines"].items():
        for k, v in m.items():
            rows.append((f"baseline: {name}", k, v, None, None))
    for k in ("precision_at_capacity", "lift_at_capacity"):
        rows.append(("capacity_top_100", k, churn["capacity"][k], None, None))
    for r in clv["summary"]:
        for k in ("PredictedMean", "ActualMean", "MAE", "RMSE"):
            rows.append(("clv_holdout", f"{r['Metric']} - {k}", r[k], None, None))
    rows.append(("clv_holdout", "top-decile share (predicted value)", clv["top_decile_share_predicted_value"], None, None))
    rows.append(("clv_holdout", "top-decile share (historical value baseline)", clv["top_decile_share_historical_value"], None, None))
    rows.append(("clv_holdout", "Spearman (predicted vs actual value)", clv["spearman_predicted_vs_actual_value"], None, None))
    pd.DataFrame(rows, columns=["Section", "Metric", "Value", "Lower95", "Upper95"]).to_csv(
        REPORTS / "final_validation_metrics.csv", index=False
    )

    c = churn["cohort"]
    base = pd.DataFrame(churn["baselines"]).T.reset_index().rename(columns={"index": "Model"})
    m4 = lambda x: f"{x:.4f}"
    ci_rows = pd.DataFrame(
        [
            {"Metric": k, "Value": churn["metrics"][k], "Lower95": churn["bootstrap_95ci"][k][0], "Upper95": churn["bootstrap_95ci"][k][1]}
            for k in churn["metrics"]
        ]
    )
    cap = churn["capacity"]
    grp = pd.DataFrame(churn["history_groups"])
    sm = pd.DataFrame(clv["summary"])
    dec = deciles.copy()

    md = f"""# Final validation metrics

Generated by `src/validate_model.py` from the **saved artifacts** in `artifacts/`.
Re-generate with `python src/validate_model.py` (or `python src/run_pipeline.py`).

## 1. Churn / no-return model - untouched temporal final test

- Snapshot: **{c['snapshot_cutoff']}** (180-day observation window ending at the cutoff; label = no valid purchase in the next 180 days)
- Eligible customers: **{c['customers']:,}**; observed no-return rate: **{pct(c['no_return_rate'])}**; mean predicted probability: **{pct(c['mean_predicted_probability'])}**
- Model: production pipeline (fit on Jun/Sep/Dec 2010 snapshots only; the June 2011 test snapshot is never used for fitting)

{md_table(ci_rows, {k: m4 for k in ["Value", "Lower95", "Upper95"]})}

95% intervals: bootstrap over test customers ({N_BOOT} resamples, seed {RANDOM_STATE}).

### Baseline comparison

{md_table(base, {k: m4 for k in ["ROC-AUC", "PR-AUC", "Brier", "LogLoss"]})}

### Capacity view (top {cap['capacity']} by calibrated probability)

- Precision: **{pct(cap['precision_at_capacity'])}** vs base rate {pct(cap['base_rate'])} (lift **{cap['lift_at_capacity']:.2f}x**)

### Performance by history group

{md_table(grp, {"NoReturnRate": pct, "ROC-AUC": m4, "PR-AUC": m4})}

### Calibration (10 uniform probability bins, final test)

{md_table(cal, {"MeanPredicted": m4, "ObservedRate": m4})}

**Interpretation.**

- The model carries useful ranking signal and beats the recency-only baseline, but it is not a
  high-certainty automated churn decision. It is a prioritisation queue, not an automatic action.
- Calibration check: predicted probabilities are on average *higher* than the observed no-return
  rate in the June 2011 test period (see the mean predicted vs observed rate above and the
  bin table). Use the probability for **ranking**; do not quote it as a literal chance of churn.
- Ranking is much weaker for single-purchase customers than for established customers
  (see the history-group table), so priorities among one-time buyers are less reliable.

## 2. CLV (BG/NBD + Gamma-Gamma) - future-holdout validation

- Fit on data up to {c['snapshot_cutoff']}; compared with realised purchases in the following {OUTCOME_DAYS} days
- Repeat customers evaluated: **{clv['customers_repeat']:,}**
- Value is **gross purchase value**, not profit

{md_table(sm, {k: (lambda x: f'{x:,.2f}') for k in ["PredictedMean", "ActualMean", "MAE", "RMSE"]})}

- Top predicted-value decile captures **{pct(clv['top_decile_share_predicted_value'])}** of realised holdout value
  (historical-value baseline: {pct(clv['top_decile_share_historical_value'])})
- Spearman rank correlation, predicted vs realised value: **{clv['spearman_predicted_vs_actual_value']:.3f}**
- The predicted-value ranking is only marginally better than simply ranking by historical
  value at the top decile; the model's main added value is the forward-looking expected
  purchase count/value, not a sharper top-decile ranking.

{md_table(dec, {"Decile": (lambda x: str(int(x))), "Customers": (lambda x: str(int(x))), "RealizedValue": (lambda x: f'{x:,.0f}'), "ShareOfRealizedValue": pct})}

## 3. How to read EVaR

`ExpectedValueAtRisk = CalibratedNoReturnProbability x ValueForRetention` mixes a **180-day risk**
with a **12-month gross-value horizon**. Treat it as a ranking score, not a revenue forecast.
"""
    (REPORTS / "final_validation_metrics.md").write_text(md)


def main():
    raw = load_transactions(DATA)
    ct = raw[raw["Customer ID"].notna()].copy()

    churn, cal = churn_validation(ct)
    clv, deciles = clv_validation(ct)
    write_reports(churn, cal, clv, deciles)

    m = churn["metrics"]
    print({"ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"], "Brier": m["Brier"], "LogLoss": m["LogLoss"]})
    print("Wrote reports/final_validation_metrics.{md,json,csv}")


if __name__ == "__main__":
    main()
