#!/usr/bin/env python3
"""
Micro-level interpretability analysis.

Target: three-class market regime (Bullish / Bearish / Neutral), defined
by the forward return with a +-3% threshold.

Model: Random Forest classifier (dominant model in the framework).

Features: daily temperature, precipitation, and season (one-hot).

Tools: SHAP (global, per-class, and grouped by proximity to a transition).
"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dependency handling
# ---------------------------------------------------------------------------
def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_ensure("scikit-learn")
_ensure("shap")

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import shap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
FORWARD_DAYS = 30
FORWARD_MAX_LOOKAHEAD = 15   # extra calendar days to find the next trading day
REGIME_THRESHOLD = 0.03
NEAR_TRANSITION_DAYS = 30
TEST_FRACTION = 0.20
CLASS_ORDER = ["Bearish", "Bullish", "Neutral"]

FEATURE_COLS = [
    "Max_Temp_C", "Min_Temp_C", "Precipitation_mm",
    "is_spring", "is_summer", "is_fall",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_main_dataset():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "dataset", "US_Agriculture_Weather_2010_2024.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Main dataset not found: {p}")

    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    n_bad = df["Date"].isna().sum()
    if n_bad > 0:
        print(f"Dropped {n_bad} rows with invalid dates")
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_transitions():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "dataset", "transition_points.csv")
    if not os.path.exists(p):
        print(f"Warning: transition_points.csv not found. Skipping near/far grouping.")
        return None
    df = pd.read_csv(p)
    df["transition_estimate"] = pd.to_datetime(df["transition_estimate"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_seasonal_features(df):
    """
    One-hot encode the season of each date.
    Winter (Dec-Jan-Feb) is the reference category (dropped).
    """
    df = df.copy()
    month = df["Date"].dt.month
    df["is_spring"] = month.isin([3, 4, 5]).astype(int)
    df["is_summer"] = month.isin([6, 7, 8]).astype(int)
    df["is_fall"]   = month.isin([9, 10, 11]).astype(int)
    return df


def build_target(df):
    """
    For each row t, find the price at the first trading day >= t + 30 calendar days.
    This handles weekends and holidays, and recovers the rows that were previously
    dropped because the exact +30 calendar date was missing.

    forward_return = (forward_price - current_price) / current_price
    regime: Bullish if R > 0.03, Bearish if R < -0.03, else Neutral.
    """
    df = df.copy()
    price_dict = dict(zip(df["Date"], df["Corn_Price_USD"]))

    forward_prices = []
    forward_returns = []
    for d in df["Date"]:
        fwd_price = None
        for offset in range(FORWARD_DAYS, FORWARD_DAYS + FORWARD_MAX_LOOKAHEAD + 1):
            fd = d + pd.Timedelta(days=offset)
            if fd in price_dict:
                fwd_price = price_dict[fd]
                break
        forward_prices.append(fwd_price)
        cur = price_dict[d]
        if fwd_price is None or cur <= 0:
            forward_returns.append(np.nan)
        else:
            forward_returns.append((fwd_price - cur) / cur)

    df["forward_price"] = forward_prices
    df["forward_return"] = forward_returns

    def _label(r):
        if pd.isna(r):
            return None
        if r > REGIME_THRESHOLD:
            return "Bullish"
        if r < -REGIME_THRESHOLD:
            return "Bearish"
        return "Neutral"

    df["regime"] = df["forward_return"].apply(_label)
    df = df.dropna(subset=["regime"]).reset_index(drop=True)
    return df


def mark_near_transitions(df, transitions):
    if transitions is None or len(transitions) == 0:
        df["is_near_transition"] = False
        return df
    df = df.copy()
    df["is_near_transition"] = False
    window = pd.Timedelta(days=NEAR_TRANSITION_DAYS)
    for t in transitions["transition_estimate"]:
        mask = (df["Date"] >= t - window) & (df["Date"] <= t + window)
        df.loc[mask, "is_near_transition"] = True
    return df


# ---------------------------------------------------------------------------
# Training / metrics
# ---------------------------------------------------------------------------
def chronological_split(df, test_fraction=TEST_FRACTION):
    n = len(df)
    n_test = int(round(n * test_fraction))
    n_train = n - n_test
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def train_rf(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_baseline(X_train, y_train):
    """Most-frequent-class baseline for reference."""
    clf = DummyClassifier(strategy="most_frequent", random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


def compute_metrics(y_true, y_pred, labels):
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(labels))),
        target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            labels[i]: {
                "precision": float(rep[labels[i]]["precision"]),
                "recall": float(rep[labels[i]]["recall"]),
                "f1": float(rep[labels[i]]["f1-score"]),
                "support": int(rep[labels[i]]["support"]),
            } for i in range(len(labels))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
        "labels": list(labels),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def _shap_per_class(explainer, X, class_names):
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return raw
    # newer shap: (n_samples, n_features, n_classes)
    return [raw[:, :, i] for i in range(raw.shape[2])]


def shap_global(clf, X, feature_names, class_names):
    explainer = shap.TreeExplainer(clf)
    per_class = _shap_per_class(explainer, X, class_names)

    overall = {}
    per_class_out = {}
    for j, fname in enumerate(feature_names):
        overall[fname] = float(np.mean([
            np.mean(np.abs(per_class[i][:, j]))
            for i in range(len(class_names))
        ]))
    for i, cname in enumerate(class_names):
        per_class_out[cname] = {
            feature_names[j]: float(np.mean(np.abs(per_class[i][:, j])))
            for j in range(len(feature_names))
        }
    return {"per_feature": overall, "per_class": per_class_out}


def shap_by_group(clf, X, groups, feature_names, class_names):
    explainer = shap.TreeExplainer(clf)
    per_class = _shap_per_class(explainer, X, class_names)
    groups = np.asarray(groups, dtype=bool)

    out = {}
    for flag, label in [(True, "near_transition"), (False, "far_from_transition")]:
        idx = np.where(groups == flag)[0]
        if len(idx) == 0:
            out[label] = {"n_samples": 0, "per_feature": {}}
            continue
        per_feature = {}
        for j, fname in enumerate(feature_names):
            per_feature[fname] = float(np.mean([
                np.mean(np.abs(per_class[i][idx, j]))
                for i in range(len(class_names))
            ]))
        out[label] = {"n_samples": int(len(idx)), "per_feature": per_feature}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("MICRO-LEVEL INTERPRETABILITY - RF CLASSIFIER")
    print("=" * 72)

    print("\n[1] Loading main dataset...")
    df = load_main_dataset()
    print(f"    Raw daily records: {len(df)}")

    print("\n[2] Adding seasonal features...")
    df = add_seasonal_features(df)
    print(f"    Added: is_spring, is_summer, is_fall (winter = reference)")

    print("\n[3] Building regime target (forward return, +-3%)...")
    df = build_target(df)
    print(f"    Rows with a valid regime: {len(df)}")

    print("\n[4] Loading transitions and marking near/far days...")
    transitions = load_transitions()
    df = mark_near_transitions(df, transitions)
    n_near = int(df["is_near_transition"].sum())
    n_far = int((~df["is_near_transition"]).sum())
    print(f"    Near transition (<{NEAR_TRANSITION_DAYS}d): {n_near}")
    print(f"    Far from transition: {n_far}")

    print("\n[5] Regime distribution:")
    for cls, cnt in df["regime"].value_counts().items():
        print(f"    {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n[6] Chronological split (80/20)...")
    df_train, df_test = chronological_split(df)
    print(f"    Train: {len(df_train)} ({df_train['Date'].min().date()} to {df_train['Date'].max().date()})")
    print(f"    Test:  {len(df_test)} ({df_test['Date'].min().date()} to {df_test['Date'].max().date()})")

    le = LabelEncoder().fit(CLASS_ORDER)
    X_train = df_train[FEATURE_COLS].values
    y_train = le.transform(df_train["regime"].values)
    X_test = df_test[FEATURE_COLS].values
    y_test = le.transform(df_test["regime"].values)

    print("\n[7] Training baseline (most-frequent class)...")
    baseline = train_baseline(X_train, y_train)
    y_base = baseline.predict(X_test)
    base_metrics = compute_metrics(y_test, y_base, CLASS_ORDER)
    print(f"    Baseline accuracy:  {base_metrics['accuracy']:.4f}")
    print(f"    Baseline F1-macro:  {base_metrics['f1_macro']:.4f}")

    print("\n[8] Training Random Forest classifier...")
    clf = train_rf(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n[9] Test-set metrics:")
    metrics = compute_metrics(y_test, y_pred, CLASS_ORDER)
    print(f"    Accuracy:     {metrics['accuracy']:.4f}  (baseline {base_metrics['accuracy']:.4f})")
    print(f"    F1-macro:     {metrics['f1_macro']:.4f}  (baseline {base_metrics['f1_macro']:.4f})")
    print(f"    F1-weighted:  {metrics['f1_weighted']:.4f}")
    print("    Per-class F1:")
    for cls in CLASS_ORDER:
        m = metrics["per_class"][cls]
        print(f"      {cls:8s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    print("\n[10] Global SHAP (full dataset, "+str(len(FEATURE_COLS))+" features)...")
    X_all = df[FEATURE_COLS].values
    sg = shap_global(clf, X_all, FEATURE_COLS, CLASS_ORDER)
    print("    Mean |SHAP| per feature:")
    for f, v in sorted(sg["per_feature"].items(), key=lambda x: -x[1]):
        print(f"      {f:20s}: {v:.4f}")

    print("\n[11] SHAP by proximity to a transition...")
    sgroup = shap_by_group(clf, X_all, df["is_near_transition"].values, FEATURE_COLS, CLASS_ORDER)
    for grp in ["near_transition", "far_from_transition"]:
        g = sgroup[grp]
        print(f"    {grp} (n={g['n_samples']}):")
        for f, v in sorted(g["per_feature"].items(), key=lambda x: -x[1]):
            print(f"      {f:20s}: {v:.4f}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\n[12] Saving outputs...")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)

    rows = []
    for cls in CLASS_ORDER:
        m = metrics["per_class"][cls]
        rows.append({"class": cls, "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "support": m["support"]})
    rows.append({"class": "MACRO", "precision": None, "recall": None,
                 "f1": metrics["f1_macro"], "support": len(y_test)})
    rows.append({"class": "WEIGHTED", "precision": None, "recall": None,
                 "f1": metrics["f1_weighted"], "support": len(y_test)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "classification_metrics.csv"), index=False)

    pd.DataFrame(metrics["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER)\
      .to_csv(os.path.join(out, "confusion_matrix.csv"))

    baseline_row = pd.DataFrame([
        {"model": "Baseline (most_frequent)", "accuracy": base_metrics["accuracy"], "f1_macro": base_metrics["f1_macro"]},
        {"model": "Random Forest",            "accuracy": metrics["accuracy"],      "f1_macro": metrics["f1_macro"]},
    ])
    baseline_row.to_csv(os.path.join(out, "baseline_comparison.csv"), index=False)

    shap_rows = [{"feature": f, "mean_abs_shap": v} for f, v in sg["per_feature"].items()]
    pd.DataFrame(shap_rows).to_csv(os.path.join(out, "shap_global.csv"), index=False)

    pc_rows = []
    for cls, feat_vals in sg["per_class"].items():
        for f, v in feat_vals.items():
            pc_rows.append({"class": cls, "feature": f, "mean_abs_shap": v})
    pd.DataFrame(pc_rows).to_csv(os.path.join(out, "shap_per_class.csv"), index=False)

    g_rows = []
    for grp, data in sgroup.items():
        for f, v in data["per_feature"].items():
            g_rows.append({"group": grp, "n_samples": data["n_samples"],
                           "feature": f, "mean_abs_shap": v})
    pd.DataFrame(g_rows).to_csv(os.path.join(out, "shap_by_group.csv"), index=False)

    df_test_out = df_test.copy()
    df_test_out["predicted_regime"] = le.inverse_transform(y_pred)
    df_test_out["correct"] = df_test_out["regime"] == df_test_out["predicted_regime"]
    df_test_out.to_csv(os.path.join(out, "test_predictions.csv"), index=False)

    summary = {
        "config": {
            "forward_days": FORWARD_DAYS,
            "forward_max_lookahead": FORWARD_MAX_LOOKAHEAD,
            "regime_threshold": REGIME_THRESHOLD,
            "near_transition_days": NEAR_TRANSITION_DAYS,
            "test_fraction": TEST_FRACTION,
            "seed": SEED,
            "features": FEATURE_COLS,
        },
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "near_transition_rows": n_near,
            "far_transition_rows": n_far,
            "class_distribution": {k: int(v) for k, v in df["regime"].value_counts().items()},
        },
        "baseline": {"accuracy": base_metrics["accuracy"], "f1_macro": base_metrics["f1_macro"]},
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "per_class": metrics["per_class"],
        },
        "shap_global": sg,
        "shap_by_group": sgroup,
    }
    with open(os.path.join(out, "micro_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"    Outputs written to: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
