#!/usr/bin/env python3
"""
Micro-level interpretability analysis.

Goal: explain which ordinary weather factors (temperature, precipitation)
drive market-regime classification, using Random Forest as the reference
model (dominant in the framework and at transition points).

Target: three-class market regime (Bullish / Bearish / Neutral), defined
by the 30-day forward return with a +-3% threshold.

Tools: SHAP (global + grouped by proximity to a transition) and LIME
(local explanations for a handful of representative days).
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
_ensure("lime")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
FORWARD_DAYS = 30
REGIME_THRESHOLD = 0.03
NEAR_TRANSITION_DAYS = 30
TEST_FRACTION = 0.20
FEATURE_COLS = ["Max_Temp_C", "Min_Temp_C", "Precipitation_mm"]
CLASS_ORDER = ["Bearish", "Bullish", "Neutral"]
N_LIME_SAMPLES = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_main_dataset():
    base = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(base, "dataset", "US_Agriculture_Weather_2010_2024.csv")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main dataset not found: {main_path}")

    df = pd.read_csv(main_path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    n_bad = df["Date"].isna().sum()
    if n_bad > 0:
        print(f"Dropped {n_bad} rows with invalid dates")
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_transitions():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "dataset", "transition_points.csv")
    if not os.path.exists(path):
        print(f"Warning: transition_points.csv not found at {path}. "
              f"near/far grouping will be skipped.")
        return None
    df = pd.read_csv(path)
    df["transition_estimate"] = pd.to_datetime(df["transition_estimate"])
    return df


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def build_target(df):
    """
    For each row t, compute:
      forward_price  = P[t + 30 days]
      forward_return = (forward_price - P[t]) / P[t]
      regime         = Bullish if R > 0.03, Bearish if R < -0.03, else Neutral

    Rows without a matching forward price are dropped.
    """
    df = df.copy()
    price_dict = dict(zip(df["Date"], df["Corn_Price_USD"]))

    forward_prices = []
    forward_returns = []
    for d in df["Date"]:
        fd = d + pd.Timedelta(days=FORWARD_DAYS)
        if fd in price_dict:
            forward_prices.append(price_dict[fd])
            cur = price_dict[d]
            if cur > 0:
                forward_returns.append((price_dict[fd] - cur) / cur)
            else:
                forward_returns.append(np.nan)
        else:
            forward_prices.append(np.nan)
            forward_returns.append(np.nan)

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
    """Add a boolean column is_near_transition to df."""
    if transitions is None or len(transitions) == 0:
        df["is_near_transition"] = False
        return df

    df = df.copy()
    df["is_near_transition"] = False
    window = pd.Timedelta(days=NEAR_TRANSITION_DAYS)

    for t in transitions["transition_estimate"]:
        mask = ((df["Date"] >= t - window) & (df["Date"] <= t + window))
        df.loc[mask, "is_near_transition"] = True
    return df


# ---------------------------------------------------------------------------
# Model training and metrics
# ---------------------------------------------------------------------------
def chronological_split(df, test_fraction=TEST_FRACTION):
    """80/20 chronological split. No shuffling. Preserves temporal order."""
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
# SHAP analysis
# ---------------------------------------------------------------------------
def compute_shap_global(clf, X, feature_names, class_names):
    """
    Returns a dict:
      per_feature: {feature: mean |SHAP| across all classes and samples}
      per_class_per_feature: {class: {feature: mean |SHAP|}}
    """
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X)

    # shap may return a list (old) or a 3D array (new). Normalize.
    if isinstance(raw, list):
        shap_per_class = raw
    else:
        # shape (n_samples, n_features, n_classes)
        shap_per_class = [raw[:, :, i] for i in range(raw.shape[2])]

    per_class = {}
    for i, cname in enumerate(class_names):
        per_class[cname] = {
            feature_names[j]: float(np.mean(np.abs(shap_per_class[i][:, j])))
            for j in range(len(feature_names))
        }

    overall = {
        feature_names[j]: float(np.mean([
            np.mean(np.abs(shap_per_class[i][:, j]))
            for i in range(len(class_names))
        ]))
        for j in range(len(feature_names))
    }

    return {"per_feature": overall, "per_class_per_feature": per_class}


def compute_shap_by_group(clf, X, groups, feature_names, class_names):
    """
    Split samples into two groups (True / False of `groups`) and compute
    mean |SHAP| per feature per group.
    """
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        shap_per_class = raw
    else:
        shap_per_class = [raw[:, :, i] for i in range(raw.shape[2])]

    groups = np.asarray(groups, dtype=bool)
    result = {}
    for flag, label in [(True, "near_transition"), (False, "far_from_transition")]:
        idx = np.where(groups == flag)[0]
        if len(idx) == 0:
            result[label] = {"n_samples": 0, "per_feature": {}}
            continue

        per_feature = {}
        for j, fname in enumerate(feature_names):
            vals = [
                np.mean(np.abs(shap_per_class[i][idx, j]))
                for i in range(len(class_names))
            ]
            per_feature[fname] = float(np.mean(vals))

        result[label] = {"n_samples": int(len(idx)), "per_feature": per_feature}

    return result


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------
def pick_lime_samples(df_test, X_test, y_test_enc, clf, class_names):
    """
    Pick representative samples for LIME:
      1. Most confident Bearish prediction
      2. Most confident Bullish prediction
      3. Most confident Neutral prediction
      4. A near-transition row with any prediction
      5. A far-from-transition row with any prediction
    Returns a list of dicts with index, description, and predicted class.
    """
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    n = len(X_test)
    picked = []
    used = set()

    def _pick(mask, desc):
        if mask is None or not mask.any():
            return
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return
        conf = probs[candidates].max(axis=1)
        order = np.argsort(-conf)
        for idx in candidates[order]:
            if idx not in used:
                used.add(idx)
                picked.append({
                    "index": int(idx),
                    "description": desc,
                    "predicted_class": class_names[preds[idx]],
                    "confidence": float(probs[idx].max()),
                })
                return

    for cls_idx, cls_name in enumerate(class_names):
        _pick(preds == cls_idx, f"confident_{cls_name}")

    if "is_near_transition" in df_test.columns:
        _pick(df_test["is_near_transition"].values, "near_transition")
        _pick(~df_test["is_near_transition"].values, "far_from_transition")

    return picked[:N_LIME_SAMPLES]


def run_lime(clf, X_train, X_test, df_test, samples, feature_names, class_names, out_dir):
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=SEED,
    )

    explanations = []
    for s in samples:
        i = s["index"]
        row = np.asarray(X_test[i], dtype=float)
        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=clf.predict_proba,
            num_features=len(feature_names),
            top_labels=len(class_names),
        )

        per_class = {}
        for cls_idx, cls_name in enumerate(class_names):
            if cls_idx in exp.local_exp:
                per_class[cls_name] = [
                    {"condition": str(cond), "weight": float(w)}
                    for cond, w in exp.local_exp[cls_idx]
                ]

        explanations.append({
            "sample_index": int(i),
            "description": s["description"],
            "predicted_class": s["predicted_class"],
            "confidence": s["confidence"],
            "date": str(df_test.iloc[i]["Date"].date()),
            "true_regime": df_test.iloc[i]["regime"],
            "feature_values": {
                f: float(X_test[i][j]) for j, f in enumerate(feature_names)
            },
            "local_explanations": per_class,
        })

    path = os.path.join(out_dir, "lime_explanations.json")
    with open(path, "w") as f:
        json.dump(explanations, f, indent=2, default=str)
    return explanations


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

    print("\n[2] Building regime target (30-day forward return, +-3%)...")
    df = build_target(df)
    print(f"    Rows with a valid regime: {len(df)}")

    print("\n[3] Loading transition points and marking near/far days...")
    transitions = load_transitions()
    df = mark_near_transitions(df, transitions)
    n_near = int(df["is_near_transition"].sum())
    n_far = int((~df["is_near_transition"]).sum())
    print(f"    Near transition (<{NEAR_TRANSITION_DAYS}d): {n_near}")
    print(f"    Far from transition: {n_far}")

    print("\n[4] Regime distribution:")
    for cls, cnt in df["regime"].value_counts().items():
        print(f"    {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n[5] Chronological split (80/20)...")
    df_train, df_test = chronological_split(df, TEST_FRACTION)
    print(f"    Train: {len(df_train)} rows "
          f"({df_train['Date'].min().date()} to {df_train['Date'].max().date()})")
    print(f"    Test:  {len(df_test)} rows "
          f"({df_test['Date'].min().date()} to {df_test['Date'].max().date()})")

    le = LabelEncoder().fit(CLASS_ORDER)
    X_train = df_train[FEATURE_COLS].values
    y_train = le.transform(df_train["regime"].values)
    X_test = df_test[FEATURE_COLS].values
    y_test = le.transform(df_test["regime"].values)

    print("\n[6] Training Random Forest classifier...")
    clf = train_rf(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n[7] Test-set metrics:")
    metrics = compute_metrics(y_test, y_pred, CLASS_ORDER)
    print(f"    Accuracy:     {metrics['accuracy']:.4f}")
    print(f"    F1-macro:     {metrics['f1_macro']:.4f}")
    print(f"    F1-weighted:  {metrics['f1_weighted']:.4f}")
    print("    Per-class F1:")
    for cls in CLASS_ORDER:
        m = metrics["per_class"][cls]
        print(f"      {cls:8s}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    print("\n[8] Global SHAP analysis (full dataset)...")
    X_all = df[FEATURE_COLS].values
    shap_global = compute_shap_global(clf, X_all, FEATURE_COLS, CLASS_ORDER)
    print("    Mean |SHAP| per feature (across all classes):")
    for f, v in sorted(shap_global["per_feature"].items(),
                        key=lambda x: -x[1]):
        print(f"      {f:20s}: {v:.4f}")

    print("\n[9] SHAP grouped by proximity to a transition...")
    shap_grouped = compute_shap_by_group(
        clf, X_all, df["is_near_transition"].values, FEATURE_COLS, CLASS_ORDER
    )
    for grp in ["near_transition", "far_from_transition"]:
        g = shap_grouped[grp]
        print(f"    {grp} (n={g['n_samples']}):")
        for f, v in sorted(g["per_feature"].items(), key=lambda x: -x[1]):
            print(f"      {f:20s}: {v:.4f}")

    print("\n[10] LIME local explanations...")
    samples = pick_lime_samples(df_test, X_test, y_test, clf, CLASS_ORDER)
    for s in samples:
        print(f"    - {s['description']}: date={df_test.iloc[s['index']]['Date'].date()}, "
              f"pred={s['predicted_class']}, conf={s['confidence']:.2f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    lime_results = run_lime(clf, X_train, X_test, df_test, samples,
                             FEATURE_COLS, CLASS_ORDER, out_dir)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\n[11] Saving outputs...")

    # Classification metrics
    metrics_rows = []
    for cls in CLASS_ORDER:
        m = metrics["per_class"][cls]
        metrics_rows.append({
            "class": cls, "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "support": m["support"],
        })
    metrics_rows.append({
        "class": "MACRO", "precision": None, "recall": None,
        "f1": metrics["f1_macro"], "support": len(y_test),
    })
    metrics_rows.append({
        "class": "WEIGHTED", "precision": None, "recall": None,
        "f1": metrics["f1_weighted"], "support": len(y_test),
    })
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(out_dir, "classification_metrics.csv"), index=False
    )

    # Confusion matrix
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=CLASS_ORDER, columns=CLASS_ORDER)
    cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # SHAP global per class
    rows = []
    for cls, feat_vals in shap_global["per_class_per_feature"].items():
        for f, v in feat_vals.items():
            rows.append({"class": cls, "feature": f, "mean_abs_shap": v})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "shap_per_class.csv"), index=False
    )

    # SHAP overall
    pd.DataFrame([
        {"feature": f, "mean_abs_shap": v}
        for f, v in shap_global["per_feature"].items()
    ]).to_csv(os.path.join(out_dir, "shap_global.csv"), index=False)

    # SHAP by group
    rows = []
    for grp, data in shap_grouped.items():
        for f, v in data["per_feature"].items():
            rows.append({
                "group": grp, "n_samples": data["n_samples"],
                "feature": f, "mean_abs_shap": v,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "shap_by_group.csv"), index=False
    )

    # Full test predictions
    df_test_out = df_test.copy()
    df_test_out["predicted_regime"] = le.inverse_transform(y_pred)
    df_test_out["correct"] = (df_test_out["regime"] == df_test_out["predicted_regime"])
    df_test_out.to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False
    )

    # Summary JSON
    summary = {
        "config": {
            "forward_days": FORWARD_DAYS,
            "regime_threshold": REGIME_THRESHOLD,
            "near_transition_days": NEAR_TRANSITION_DAYS,
            "test_fraction": TEST_FRACTION,
            "seed": SEED,
        },
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "near_transition_rows": n_near,
            "far_transition_rows": n_far,
            "class_distribution": {
                k: int(v) for k, v in df["regime"].value_counts().items()
            },
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "per_class": metrics["per_class"],
        },
        "shap_global": shap_global,
        "shap_by_group": shap_grouped,
        "lime_samples": [s["description"] for s in samples],
    }
    with open(os.path.join(out_dir, "micro_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"    Outputs written to: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
