"""Customer Retention Intelligence - Streamlit dashboard.

Reads precomputed artifacts only (artifacts/, reports/). It never trains models.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
OBS_DAYS = 180

# Presentation bands on the calibrated 180-day no-return probability.
# They only make the table easier to scan; they are NOT decision thresholds.
RISK_BANDS = [(0.0, 0.25, "Low"), (0.25, 0.50, "Medium"), (0.50, 1.01, "High")]

st.set_page_config(page_title="Customer Retention Intelligence", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_scores():
    s = pd.read_csv(ART / "customer_scores.csv", parse_dates=["SnapshotCutoff"])
    s["Customer ID"] = s["Customer ID"].astype(int)
    s["RiskBand"] = pd.cut(
        s["CalibratedNoReturnProbability"],
        bins=[b[0] for b in RISK_BANDS] + [RISK_BANDS[-1][1]],
        labels=[b[2] for b in RISK_BANDS],
        right=False,
        include_lowest=True,
    ).astype(str)
    s = s.sort_values("ExpectedValueAtRisk", ascending=False).reset_index(drop=True)
    s["EVaRRank"] = s.index + 1
    return s


@st.cache_data
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def gbp(x):
    return f"£{x:,.0f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def display_table(df):
    """Human-friendly view (percent / £ formatting). Raw numbers stay in downloads."""
    out = pd.DataFrame(
        {
            "Rank": df["EVaRRank"],
            "Customer ID": df["Customer ID"],
            "No-return risk": df["CalibratedNoReturnProbability"].map(pct),
            "Risk band": df["RiskBand"],
            "Value for retention": df["ValueForRetention"].map(gbp),
            "EVaR (ranking score)": df["ExpectedValueAtRisk"].map(gbp),
            "Value basis": df["CLVStatus"].str.replace("Not Eligible - ", "Historical value: ", regex=False)
            .str.replace("Predicted CLV Eligible", "12-month predicted CLV", regex=False),
            "Priority": df["RetentionPriority"],
        }
    )
    return out


# ----------------------------------------------------------------------------- guards
if not (ART / "customer_scores.csv").exists():
    st.error("Run the offline pipeline first: `python src/run_pipeline.py`")
    st.stop()

scores = load_scores()
meta = load_json(ART / "model_metadata.json") or {}
manifest = load_json(ART / "manifest.json")
validation = load_json(REPORTS / "final_validation_metrics.json")

cutoff = scores["SnapshotCutoff"].iloc[0]
obs_start = cutoff - pd.Timedelta(days=OBS_DAYS)
n_priority = int((scores.RetentionPriority == "Priority").sum())

st.title("Customer Retention Intelligence")
st.caption(
    f"Data as of **{cutoff:%d %b %Y}** · scored population: **{len(scores):,} customers** with at least one "
    f"valid purchase in the {OBS_DAYS} days ending {cutoff:%d %b %Y} (after {obs_start:%d %b %Y}) · "
    "customers with no purchase in that window are not scored."
)
st.warning(
    "**How to read EVaR.** Expected Value at Risk = 180-day no-return probability × value for retention. "
    "The risk is a **180-day** outcome while the value is a **12-month gross purchase value** (not profit), "
    "so EVaR is a **ranking score for prioritisation, not a revenue forecast**.",
    icon="⚠️",
)

tab_over, tab_lookup, tab_queue, tab_val, tab_method = st.tabs(
    ["Executive Overview", "Customer Lookup", "Retention Queue", "Model Validation", "Methodology"]
)

# ----------------------------------------------------------------------------- overview
with tab_over:
    total_evar = scores.ExpectedValueAtRisk.sum()
    prio_evar = scores.loc[scores.RetentionPriority == "Priority", "ExpectedValueAtRisk"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored customers", f"{len(scores):,}")
    c2.metric("Priority customers", f"{n_priority:,}")
    c3.metric("Total EVaR (ranking score)", gbp(total_evar))
    c4.metric("Share of EVaR in priority queue", pct(prio_evar / total_evar))

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Risk bands")
        bands = (
            scores.groupby("RiskBand")
            .agg(Customers=("Customer ID", "size"), EVaR=("ExpectedValueAtRisk", "sum"))
            .reindex([b[2] for b in RISK_BANDS])
            .reset_index()
        )
        bands["EVaR"] = bands["EVaR"].map(gbp)
        st.dataframe(bands, hide_index=True, use_container_width=True)
        st.caption("Bands: Low < 25%, Medium 25-50%, High ≥ 50% predicted no-return probability (display only).")
    with right:
        st.subheader("Top 20 customers by EVaR")
        st.dataframe(display_table(scores.head(20)), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- lookup
with tab_lookup:
    cid = st.number_input("Customer ID", min_value=0, value=int(scores["Customer ID"].iloc[0]), step=1)
    row = scores[scores["Customer ID"] == cid]
    if row.empty:
        st.info(
            f"Customer {cid} is not in the scored population. Only customers with at least one valid "
            f"purchase in the {OBS_DAYS} days ending {cutoff:%d %b %Y} are scored."
        )
    else:
        r = row.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("No-return risk (180d)", pct(r.CalibratedNoReturnProbability), r.RiskBand, delta_color="off")
        b.metric("Value for retention", gbp(r.ValueForRetention))
        c.metric("EVaR (ranking score)", gbp(r.ExpectedValueAtRisk))
        d.metric("Queue rank", f"{int(r.EVaRRank):,} of {len(scores):,}", r.RetentionPriority, delta_color="off")
        basis = (
            "12-month predicted gross purchase CLV (repeat customer)"
            if r.CLVStatus == "Predicted CLV Eligible"
            else "historical purchase value (one-time customer, no CLV prediction possible)"
        )
        st.markdown(
            f"**Why this rank:** a **{pct(r.CalibratedNoReturnProbability)}** estimated chance of no purchase in the "
            f"next 180 days × **{gbp(r.ValueForRetention)}** value ({basis}) = **{gbp(r.ExpectedValueAtRisk)}** EVaR. "
            f"{'This customer is in the priority queue.' if r.RetentionPriority == 'Priority' else 'This customer is outside the priority queue (capacity ' + str(n_priority) + ').'}"
        )
        with st.expander("Raw record"):
            st.dataframe(row.drop(columns=["EVaRRank"]).astype(str).T.rename(columns={row.index[0]: "value"}), use_container_width=True)

# ----------------------------------------------------------------------------- queue
with tab_queue:
    f1, f2, f3 = st.columns(3)
    n = f1.slider("Queue size", 1, min(500, len(scores)), min(n_priority, len(scores)))
    bands_sel = f2.multiselect("Risk band", [b[2] for b in RISK_BANDS], default=[b[2] for b in RISK_BANDS])
    basis_sel = f3.multiselect("Value basis", sorted(scores.CLVStatus.unique()), default=sorted(scores.CLVStatus.unique()))
    pool = scores[scores.RiskBand.isin(bands_sel) & scores.CLVStatus.isin(basis_sel)]
    queue = pool.head(n)
    st.write(f"Showing **{len(queue):,}** customers, ranked by EVaR · total EVaR {gbp(queue.ExpectedValueAtRisk.sum())}")
    st.dataframe(display_table(queue), hide_index=True, use_container_width=True)
    st.download_button(
        "Download queue (CSV, raw values)",
        queue.drop(columns=["EVaRRank"]).to_csv(index=False).encode(),
        "retention_queue.csv",
        "text/csv",
    )

# ----------------------------------------------------------------------------- validation
with tab_val:
    if not validation:
        st.error("reports/final_validation_metrics.json not found. Run `python src/run_pipeline.py`.")
    else:
        ch, clv = validation["churn"], validation["clv"]
        co = ch["cohort"]
        st.subheader("Churn / no-return model - untouched temporal final test")
        st.caption(
            f"Snapshot {co['snapshot_cutoff']} · {co['customers']:,} customers · observed no-return rate "
            f"{pct(co['no_return_rate'])} · mean predicted {pct(co['mean_predicted_probability'])}. "
            "Model fitted on Jun/Sep/Dec 2010 snapshots only."
        )
        cols = st.columns(4)
        for col, k in zip(cols, ["ROC-AUC", "PR-AUC", "Brier", "LogLoss"]):
            lo, hi = ch["bootstrap_95ci"][k]
            col.metric(k, f"{ch['metrics'][k]:.4f}", f"95% CI {lo:.3f}-{hi:.3f}", delta_color="off")

        st.markdown("**Baseline comparison**")
        base = pd.DataFrame(ch["baselines"]).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(base.round(4), hide_index=True, use_container_width=True)

        cap = ch["capacity"]
        st.markdown(
            f"**Capacity view:** top {cap['capacity']} customers by probability have precision "
            f"**{pct(cap['precision_at_capacity'])}** vs base rate {pct(cap['base_rate'])} "
            f"(lift **{cap['lift_at_capacity']:.2f}×**)."
        )
        st.markdown("**By customer history**")
        st.dataframe(pd.DataFrame(ch["history_groups"]).round(4), hide_index=True, use_container_width=True)

        cal_path = REPORTS / "calibration_final_test.csv"
        if cal_path.exists():
            st.markdown("**Calibration (final test)** - predicted vs observed no-return rate by probability bin")
            cal = pd.read_csv(cal_path)
            st.line_chart(cal.set_index("MeanPredicted")[["ObservedRate"]].assign(Perfect=cal.set_index("MeanPredicted").index))
            st.dataframe(cal.round(4), hide_index=True, use_container_width=True)
        st.info(
            "Interpretation: useful ranking signal, not a high-certainty churn decision. Predicted probabilities run "
            "higher than the observed rate in the test period - use them to rank, not as literal probabilities."
        )

        st.subheader("CLV - future-holdout validation (repeat customers)")
        st.caption(f"{clv['customers_repeat']:,} repeat customers · 180-day holdout after {co['snapshot_cutoff']} · gross purchase value, not profit")
        st.dataframe(pd.DataFrame(clv["summary"]).round(2), hide_index=True, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Top-decile share of realised value", pct(clv["top_decile_share_predicted_value"]))
        m2.metric("Historical-value baseline", pct(clv["top_decile_share_historical_value"]))
        m3.metric("Spearman (pred. vs actual value)", f"{clv['spearman_predicted_vs_actual_value']:.3f}")
        st.caption("Full evidence: `reports/final_validation_metrics.md` (also .json / .csv) and the executed `notebooks/analysis_final.ipynb`.")

# ----------------------------------------------------------------------------- methodology
with tab_method:
    st.markdown(
        f"""
**Target:** `{meta.get('target', 'NoPurchaseNext180Days')}` - at least one valid purchase in the {OBS_DAYS}-day observation
window, then **no valid purchase in the following {OBS_DAYS} days** (an operational non-return proxy, not permanent churn).

**Temporal design:** train {', '.join(meta.get('train_cutoffs', []))}; validation {meta.get('validation_cutoff', '')};
diagnostic Mar 2011; final test Jun 2011. Production refit: **{meta.get('production_refit', 'train+validation only')}**
(final test excluded: {meta.get('final_test_excluded', True)}).

**Model:** log1p on order count / average order value → logistic regression → sigmoid calibration (5-fold).

**Value:** repeat customers use a 12-month `PredictedGrossPurchaseCLV` (BG/NBD + Gamma-Gamma, 1% monthly discount);
one-time customers use their historical purchase value.

**Priority queue:** top {n_priority} customers by EVaR (capacity assumption, configurable offline).
"""
    )
    st.info("EVaR mixes a 180-day risk with a 12-month gross-value horizon: it is a prioritisation score, not a revenue forecast.")
    if manifest:
        st.subheader("Artifact provenance")
        env = manifest["environment"]
        st.markdown(
            f"- Generated: `{manifest['generated_utc']}` with `{manifest['generation_command']}`\n"
            f"- Python `{env['python']}` · pandas `{env['packages'].get('pandas')}` · scikit-learn `{env['packages'].get('scikit-learn')}` · "
            f"lifetimes `{env['packages'].get('lifetimes')}` · environment matches `requirements.txt`: **{env['matches_requirements_txt']}**\n"
            f"- Source data sha256: `{manifest['source_data']['sha256'][:16]}…`\n"
            f"- {len(manifest['outputs'])} output files hashed in `artifacts/manifest.json` (verify with `python src/verify_artifacts.py`)"
        )
