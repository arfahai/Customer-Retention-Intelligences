"""Fit the production churn / no-return model.

Production fit = train snapshots (Jun + Sep 2010) + validation snapshot (Dec 2010).
The final-test snapshot (9 Jun 2011) is never used for fitting.

Output: artifacts/churn_pipeline.joblib, artifacts/model_metadata.json
"""
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import (
    FEATURES,
    LOG_COLS,
    TARGET,
    TRAIN_CUTOFFS,
    VALIDATION_CUTOFF,
    load_transactions,
    prepare_snapshot,
)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_II.xlsx"
ART = ROOT / "artifacts"


def build_pipeline():
    return Pipeline(
        [
            (
                "feature_transform",
                ColumnTransformer(
                    [("log1p", FunctionTransformer(np.log1p, validate=False), LOG_COLS)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def development_frame(ct):
    """Train + validation snapshots, in time order."""
    cutoffs = TRAIN_CUTOFFS + [VALIDATION_CUTOFF]
    return pd.concat([prepare_snapshot(c, ct) for c in cutoffs], ignore_index=True)


def main():
    raw = load_transactions(DATA)
    ct = raw[raw["Customer ID"].notna()].copy()

    dev = development_frame(ct)

    model = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)
    model.fit(dev[FEATURES], dev[TARGET])

    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "churn_pipeline.joblib")

    metadata = {
        "target": TARGET,
        "features": FEATURES,
        "log1p_features": LOG_COLS,
        "production_refit": "train+validation only",
        "final_test_excluded": True,
        "train_cutoffs": [str(c.date()) for c in TRAIN_CUTOFFS],
        "validation_cutoff": str(VALIDATION_CUTOFF.date()),
        "n_training_rows": int(len(dev)),
        "training_positive_rate": round(float(dev[TARGET].mean()), 6),
        "calibration": {"method": "sigmoid", "cv": 5},
        "random_state": RANDOM_STATE,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    (ART / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved churn_pipeline.joblib")


if __name__ == "__main__":
    main()
