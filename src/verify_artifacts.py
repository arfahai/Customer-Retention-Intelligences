"""Verify that committed artifacts are one coherent set.

Default checks (fast, no data load):
  * every file in artifacts/manifest.json still has its recorded sha256
  * environment matches requirements.txt pins
  * churn_scores.csv and customer_scores.csv agree; 2,626-style population integrity;
    priority queue == top-N EVaR

--full additionally re-scores the whole population from the saved model + CLV file and
requires customer_scores.csv / churn_scores.csv to be reproduced exactly (needs the dataset).

Exit code 0 = all checks passed, 1 = at least one failed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from make_manifest import ART, DATA, ROOT, installed_packages, requirements_pins, sha256

RETENTION_CAPACITY = 100
failures = []


def check(ok, label):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


def verify_hashes(manifest):
    for rel, entry in manifest["outputs"].items():
        p = ROOT / rel
        check(p.exists() and sha256(p) == entry["sha256"], f"output hash: {rel}")
    for rel, digest in manifest["code"].items():
        p = ROOT / rel
        check(p.exists() and sha256(p) == digest, f"code hash: {rel}")
    check(
        DATA.exists() and sha256(DATA) == manifest["source_data"]["sha256"],
        "source data hash: data/online_retail_II.xlsx",
    )


def verify_environment(manifest):
    pins, installed = requirements_pins(), installed_packages()
    mism = {k: (installed.get(k), v) for k, v in pins.items() if k in installed and installed[k] != v}
    check(not mism, f"installed packages match requirements.txt {mism if mism else ''}".strip())
    check(
        manifest["environment"]["packages"] == installed,
        "installed packages equal the versions recorded in the manifest",
    )


def verify_consistency():
    cs = pd.read_csv(ART / "customer_scores.csv")
    ch = pd.read_csv(ART / "churn_scores.csv")
    clv = pd.read_csv(ART / "clv_scores.csv")
    check(cs["Customer ID"].is_unique, "customer_scores: unique Customer ID")
    check(len(cs) == len(ch), f"customer_scores and churn_scores same row count ({len(cs)})")
    check(
        np.array_equal(cs["Customer ID"].to_numpy(), ch["Customer ID"].to_numpy())
        and np.array_equal(
            cs["CalibratedNoReturnProbability"].to_numpy(),
            ch["CalibratedNoReturnProbability"].to_numpy(),
        ),
        "customer_scores and churn_scores carry identical probabilities",
    )
    check(cs["Customer ID"].isin(clv["Customer ID"]).all(), "every scored customer has CLV/value row")
    check(cs["ValueForRetention"].notna().all(), "no customer without ValueForRetention")
    top = set(cs.nlargest(RETENTION_CAPACITY, "ExpectedValueAtRisk")["Customer ID"])
    pri = set(cs.loc[cs.RetentionPriority.eq("Priority"), "Customer ID"])
    check(top == pri and len(pri) == RETENTION_CAPACITY, f"Priority queue == top {RETENTION_CAPACITY} EVaR")
    check(
        np.allclose(cs.ExpectedValueAtRisk, cs.CalibratedNoReturnProbability * cs.ValueForRetention, rtol=1e-12, atol=0),
        "EVaR == probability x value",
    )


def verify_full():
    import joblib

    from clv import fit_clv_models  # noqa: F401  (import check only)
    from common import FEATURES, PRODUCTION_CUTOFF, load_transactions
    from score_customers import score_population

    raw = load_transactions(DATA)
    ct = raw[raw["Customer ID"].notna()].copy()
    model = joblib.load(ART / "churn_pipeline.joblib")
    clv = pd.read_csv(ART / "clv_scores.csv")
    churn, out = score_population(ct, model, clv)

    saved = pd.read_csv(ART / "customer_scores.csv")
    saved_churn = pd.read_csv(ART / "churn_scores.csv")
    check(
        np.abs(out.CalibratedNoReturnProbability.to_numpy() - saved.CalibratedNoReturnProbability.to_numpy()).max() < 1e-12,
        "FULL: saved model re-scores to customer_scores.csv (max prob diff < 1e-12)",
    )
    check(
        np.abs(out.ExpectedValueAtRisk.to_numpy() - saved.ExpectedValueAtRisk.to_numpy()).max() < 1e-6,
        "FULL: EVaR reproduced",
    )
    check(
        np.abs(churn[FEATURES].to_numpy() - saved_churn[FEATURES].to_numpy()).max() < 1e-9,
        "FULL: churn_scores feature columns reproduced",
    )


def main():
    full = "--full" in sys.argv
    manifest_path = ART / "manifest.json"
    check(manifest_path.exists(), "artifacts/manifest.json exists")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_hashes(manifest)
        verify_environment(manifest)
    verify_consistency()
    if full:
        verify_full()
    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
