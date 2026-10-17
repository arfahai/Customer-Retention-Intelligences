import numpy as np
import pandas as pd


OBSERVATION_DAYS = 180
OUTCOME_DAYS = 180

# ---- Single source of truth shared by every pipeline script -----------------
# Temporal design (see README section 6)
TRAIN_CUTOFFS = [pd.Timestamp("2010-06-09"), pd.Timestamp("2010-09-09")]
VALIDATION_CUTOFF = pd.Timestamp("2010-12-09")
PRODUCTION_CUTOFF = pd.Timestamp("2011-06-09")  # also the final-test snapshot
RETENTION_CAPACITY = 100

FEATURES = [
    "RecencyDays",
    "PurchaseOrderCount",
    "AvgOrderValue",
    "TenureDays",
    "MedianPurchaseGap",
    "SinglePurchaseHistory",
    "CancellationOrderCount",
    "CancellationRate",
    "CancellationValueRatio",
]
LOG_COLS = ["PurchaseOrderCount", "AvgOrderValue"]
GAP_COLS = ["MedianPurchaseGap", "MeanPurchaseGap", "PurchaseGapCount"]
TARGET = "NoPurchaseNext180Days"


def load_transactions(path):
    a = pd.read_excel(path, sheet_name=0)
    b = pd.read_excel(path, sheet_name=1)

    r = pd.concat([a, b], ignore_index=True)
    r["InvoiceDate"] = pd.to_datetime(r["InvoiceDate"])

    r = r.drop_duplicates().copy()

    r["Is_C_Invoice"] = (
        r["Invoice"]
        .astype(str)
        .str.startswith("C", na=False)
    )

    r["Transaction_Type"] = np.select(
        [
            r["Is_C_Invoice"],
            (r["Quantity"] > 0) & (r["Price"] > 0),
        ],
        [
            "Cancellation",
            "Valid Purchase",
        ],
        default="Other",
    )

    return r


def build_snapshot(cutoff, ct):
    start = cutoff - pd.Timedelta(days=180)
    end = cutoff + pd.Timedelta(days=180)

    obs = ct[
        (ct.InvoiceDate > start) &
        (ct.InvoiceDate <= cutoff)
    ]

    fut = ct[
        (ct.InvoiceDate > cutoff) &
        (ct.InvoiceDate <= end)
    ]

    p = obs[
        obs.Transaction_Type == "Valid Purchase"
    ].copy()

    fp = fut[
        fut.Transaction_Type == "Valid Purchase"
    ].copy()

    o = (
        p.assign(
            OrderValue=p.Quantity * p.Price
        )
        .groupby(
            ["Customer ID", "Invoice"],
            as_index=False
        )
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            OrderValue=("OrderValue", "sum"),
        )
    )

    s = pd.concat(
        [
            o.groupby("Customer ID")
            .InvoiceDate
            .max()
            .rename("LastPurchaseDate"),

            o.groupby("Customer ID")
            .size()
            .rename("PurchaseOrderCount"),

            o.groupby("Customer ID")
            .OrderValue
            .mean()
            .rename("AvgOrderValue"),

            o.groupby("Customer ID")
            .InvoiceDate
            .min()
            .rename("FirstPurchaseDate"),
        ],
        axis=1,
    ).reset_index()

    g = o.sort_values(
        ["Customer ID", "InvoiceDate"]
    ).copy()

    g["gap"] = (
        g.groupby("Customer ID")
        .InvoiceDate
        .diff()
        .dt.total_seconds()
        / 86400
    )

    gs = (
        g.dropna(subset=["gap"])
        .groupby("Customer ID")
        .gap
        .agg(
            MedianPurchaseGap="median",
            MeanPurchaseGap="mean",
            PurchaseGapCount="count",
        )
    )

    s = s.merge(
        gs.reset_index(),
        on="Customer ID",
        how="left",
    )

    s["RecencyDays"] = (
        cutoff - s.LastPurchaseDate
    ).dt.total_seconds() / 86400

    s["TenureDays"] = (
        cutoff - s.FirstPurchaseDate
    ).dt.total_seconds() / 86400

    s = s.merge(
        fp.groupby("Customer ID")
        .Invoice
        .nunique()
        .rename("FuturePurchaseOrderCount")
        .reset_index(),
        on="Customer ID",
        how="left",
    )

    s["FuturePurchaseOrderCount"] = (
        s.FuturePurchaseOrderCount.fillna(0)
    )

    s["NoPurchaseNext180Days"] = (
        s.FuturePurchaseOrderCount == 0
    ).astype(int)

    s["SinglePurchaseHistory"] = (
        s.PurchaseOrderCount == 1
    ).astype(int)

    s["SnapshotCutoff"] = cutoff

    return s


def build_cancellation_features(cutoff, ct):
    start = cutoff - pd.Timedelta(days=180)

    o = ct[
        (ct.InvoiceDate > start) &
        (ct.InvoiceDate <= cutoff)
    ]

    c = o[
        o.Transaction_Type == "Cancellation"
    ].copy()

    p = o[
        o.Transaction_Type == "Valid Purchase"
    ].copy()

    cc = (
        c.groupby("Customer ID")
        .Invoice
        .nunique()
        .rename("CancellationOrderCount")
    )

    pc = (
        p.groupby("Customer ID")
        .Invoice
        .nunique()
        .rename("PC")
    )

    c["v"] = c.Quantity * c.Price
    p["v"] = p.Quantity * p.Price

    cv = (
        c.groupby("Customer ID")
        .v
        .sum()
        .rename("CV")
    )

    pv = (
        p.groupby("Customer ID")
        .v
        .sum()
        .rename("PV")
    )

    f = pd.concat(
        [cc, pc, cv, pv],
        axis=1,
    ).fillna(0)

    f["CancellationRate"] = np.where(
        f.PC > 0,
        f.CancellationOrderCount / f.PC,
        0,
    )

    f["CancellationValueRatio"] = np.where(
        f.PV > 0,
        f.CV.abs() / f.PV,
        0,
    )

    return f[
        [
            "CancellationOrderCount",
            "CancellationRate",
            "CancellationValueRatio",
        ]
    ].reset_index()


def attach_cancellation_features(s, ct):
    out = []

    for c, p in s.groupby("SnapshotCutoff"):
        x = p.merge(
            build_cancellation_features(c, ct),
            on="Customer ID",
            how="left",
        )

        x[
            [
                "CancellationOrderCount",
                "CancellationRate",
                "CancellationValueRatio",
            ]
        ] = x[
            [
                "CancellationOrderCount",
                "CancellationRate",
                "CancellationValueRatio",
            ]
        ].fillna(0)

        out.append(x)

    return pd.concat(
        out,
        ignore_index=True,
    )


def prepare_snapshot(cutoff, ct):
    """Snapshot + cancellation features + gap NaN handling (one-time customers)."""
    s = attach_cancellation_features(build_snapshot(cutoff, ct), ct)
    s[GAP_COLS] = s[GAP_COLS].fillna(0)
    return s
