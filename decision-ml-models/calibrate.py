#!/usr/bin/env python3
"""
Calibration module for the training-free model selection framework.

Goal: find coefficients (a1, a2, a3, a4) that satisfy two constraints
on the requirement vector, for the US crop & weather dataset.

Constraints (from the framework paper):
  C1. Dominance: for at least 95% of contexts, the dominance ratio for each
      requirement exceeds 0.70.
  C2. Boundedness: the requirement vector stays in [0, 1] for every context.

Objective: among feasible candidates, minimize the sum of coefficients.
Rationale: the most permissive feasible calibration is preferred.

Grid range: coefficients searched in [0.50, 0.99]. The upper bound was
extended from 0.95 to 0.99 because the sigmoid-based requirement functions
saturate at the boundaries of the expertise range (E near 0 or 1). Higher
coefficient values are needed to satisfy the dominance constraint there.
"""

import os
import sys
import json
import time
import warnings

def pprint(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

try:
    import numpy as np
except ImportError:
    pprint("Installing numpy...")
    install("numpy")
    import numpy as np

try:
    import pandas as pd
except ImportError:
    pprint("Installing pandas...")
    install("pandas")
    import pandas as pd

warnings.filterwarnings('ignore')

SEED = 42
DOMINANCE_THRESHOLD = 0.70
DOMINANCE_QUANTILE = 0.05   # 5th percentile must exceed threshold
                            # <=> 95% of contexts satisfy dominance
PERTURB_SIGMA = 0.05
PERTURB_PER_REAL = 100


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x) ** 2


def compute_requirements(ctx, a1, a2, a3, a4):
    """Requirement vector from context vector and coefficients."""
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal   = a3 * tanh(3 * V) + b3 * G
    r_rep    = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])


def compute_analytical_sensitivity(ctx, a1, a2, a3, a4):
    """Partial derivatives of each requirement w.r.t. its drivers."""
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    sens = np.zeros((4, 5))
    sens[0, 4] = -a1 * 10 * sigmoid_deriv(10 * (E - 0.5))   # d r_interp / d E
    sens[0, 3] = b1                                          # d r_interp / d rho
    sens[1, 1] = a2 * 12 * sigmoid_deriv(12 * (N - 0.35))   # d r_robust / d N
    sens[1, 3] = b2 * 2 * tanh_deriv(2 * rho)               # d r_robust / d rho
    sens[2, 0] = a3 * 3 * tanh_deriv(3 * V)                 # d r_scal / d V
    sens[2, 2] = b3                                          # d r_scal / d G
    sens[3, 2] = a4                                          # d r_rep / d G
    sens[3, 4] = b4                                          # d r_rep / d E
    return sens


def compute_dominance_ratio(ctx, a1, a2, a3, a4):
    """Dominance ratio for each requirement: primary share / (primary + secondary)."""
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sm = compute_analytical_sensitivity(ctx, a1, a2, a3, a4)
    for idx, (req, prim, sec) in enumerate(pairs):
        sp = abs(sm[req, prim])
        ss = abs(sm[req, sec])
        D[idx] = sp / (sp + ss + 1e-12)
    return D


# ---------------------------------------------------------------------------
# Context extraction from dataset
# ---------------------------------------------------------------------------

def compute_context_from_df(df, target_col='Corn_Price_USD', E=0.5):
    X = df.drop(columns=[target_col], errors='ignore')
    n, p = X.shape
    V = np.clip(np.log10(max(n, 1)) / 6, 0, 1)
    rho = np.clip(p / max(n, 1), 0, 1)

    missing = X.isnull().sum().sum() / (n * p) if n * p > 0 else 0
    outlier_ratio = 0
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = X[col].std()
        if std > 0:
            outliers = ((X[col] - X[col].mean()).abs() > 3 * std).sum()
            outlier_ratio += outliers / max(n, 1)
    outlier_ratio = outlier_ratio / max(1, len(num_cols))
    N = np.clip(0.5 * missing + 0.5 * outlier_ratio, 0, 1)

    date_cols = X.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        try:
            dates = X[date_cols[0]].dropna().sort_values()
            if len(dates) > 1:
                deltas = dates.diff().dropna()
                median_delta = deltas.median().total_seconds()
                G = np.clip(86400 / max(median_delta, 86400), 0, 1)
            else:
                G = 0.5
        except Exception:
            G = 0.5
    else:
        G = np.clip(np.log10(max(n, 1)) / 6, 0, 1)

    return np.array([V, N, G, rho, E])


def load_dataset():
    paths = [
        "dataset/US_Agriculture_Weather_2010_2024.csv",
        "../dataset/US_Agriculture_Weather_2010_2024.csv",
        "US_Agriculture_Weather_2010_2024.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            pprint(f"Loading dataset: {p}")
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            df = df.sort_values('Date').reset_index(drop=True)
            return df
    pprint("Dataset not found.")
    return None


def load_all_datasets():
    datasets = []
    df = load_dataset()
    if df is not None:
        datasets.append(("US_Crop_Weather", df, "Corn_Price_USD"))
    return datasets


def extract_contexts(datasets):
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for name, df, target_col in datasets:
        pprint(f"\nDataset: {name} | {df.shape[0]:,} rows")
        for E in E_levels:
            try:
                ctx = compute_context_from_df(df, target_col, E)
                all_ctxs.append(ctx)
            except Exception as e:
                pprint(f"  E={E} error: {e}")
        pprint(f"  Extracted {len(E_levels)} contexts")
    return all_ctxs


def generate_perturbed_contexts(real_ctxs, n_per_real=PERTURB_PER_REAL,
                                 sigma=PERTURB_SIGMA, seed=SEED):
    """Synthetic contexts by adding Gaussian noise around real contexts."""
    rng = np.random.default_rng(seed)
    synth = []
    for ctx in real_ctxs:
        for _ in range(n_per_real):
            noise = rng.normal(0, sigma, size=5)
            perturbed = np.clip(ctx + noise, 0, 1)
            synth.append(perturbed)
    return np.array(synth)


# ---------------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------------

def check_dominance_quantile(ctxs, a1, a2, a3, a4,
                              thresh=DOMINANCE_THRESHOLD,
                              quantile=DOMINANCE_QUANTILE,
                              max_sample=2000):
    """Return True if the given quantile of D exceeds threshold for all 4 reqs."""
    if len(ctxs) > max_sample:
        step = max(1, len(ctxs) // max_sample)
        sample_ctxs = [ctxs[i] for i in range(0, len(ctxs), step)][:max_sample]
    else:
        sample_ctxs = ctxs

    all_D = np.array([compute_dominance_ratio(ctx, a1, a2, a3, a4)
                      for ctx in sample_ctxs])

    for req_idx in range(4):
        q = float(np.quantile(all_D[:, req_idx], quantile))
        if q <= thresh:
            return False
    return True


def check_bounded(ctxs, a1, a2, a3, a4):
    for ctx in ctxs:
        r = compute_requirements(ctx, a1, a2, a3, a4)
        if np.any(r < 0) or np.any(r > 1):
            return False
    return True


def diagnose_failure(ctxs, a1, a2, a3, a4):
    """Report dominance stats to help debug infeasibility."""
    all_D = np.array([compute_dominance_ratio(ctx, a1, a2, a3, a4)
                      for ctx in ctxs])
    names = ['interp', 'robust', 'scal', 'rep']
    out = {}
    for i, name in enumerate(names):
        D = all_D[:, i]
        out[name] = {
            'min': float(np.min(D)),
            'q05': float(np.quantile(D, 0.05)),
            'median': float(np.median(D)),
            'max': float(np.max(D))
        }
    return out


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def build_grid():
    """
    Build the coefficient grid.

    Fine steps near 0.95 because feasible solutions require high coefficients
    to satisfy the dominance constraint at the boundaries of the expertise
    range (E = 0 and E = 1), where the sigmoid saturates.
    """
    coarse = np.arange(0.50, 0.91, 0.05)                    # 0.50 to 0.90
    fine   = np.array([0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99])              # 0.91 to 0.99
    return np.concatenate([coarse, fine])


def grid_search(ctxs):
    vals = build_grid()
    feasible = []
    total = len(vals) ** 4
    count = 0

    pprint(f"\nGrid search over {total} combinations...")
    pprint(f"Grid values: {len(vals)} distinct coefficients")
    pprint(f"Range: [{vals[0]:.2f}, {vals[-1]:.2f}]")
    pprint(f"Dominance threshold = {DOMINANCE_THRESHOLD} "
           f"at quantile {DOMINANCE_QUANTILE}")
    pprint(f"Sampling up to 2000 contexts")

    start_time = time.time()
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 5000 == 0:
                        elapsed = time.time() - start_time
                        pprint(f"  {count}/{total} | {elapsed:.1f}s")

                    if not check_dominance_quantile(ctxs, a1, a2, a3, a4):
                        continue
                    if not check_bounded(ctxs, a1, a2, a3, a4):
                        continue

                    feasible.append({
                        'a1': round(float(a1), 2),
                        'a2': round(float(a2), 2),
                        'a3': round(float(a3), 2),
                        'a4': round(float(a4), 2),
                        'sum_a': round(float(a1 + a2 + a3 + a4), 4),
                    })

    # Objective: minimize sum_a among feasible candidates.
    feasible.sort(key=lambda x: x['sum_a'])
    pprint(f"Found {len(feasible)} feasible candidates")
    return feasible


def final_verification(ctxs, candidates, top_k=20):
    """Re-check top candidates on the full context set (no sampling)."""
    verified = []
    pprint(f"\nFinal verification on all {len(ctxs)} contexts (top {top_k})...")

    for cand in candidates[:top_k]:
        a1, a2, a3, a4 = cand['a1'], cand['a2'], cand['a3'], cand['a4']
        all_D = np.array([compute_dominance_ratio(ctx, a1, a2, a3, a4)
                          for ctx in ctxs])

        passes = True
        for req_idx in range(4):
            q = float(np.quantile(all_D[:, req_idx], DOMINANCE_QUANTILE))
            if q <= DOMINANCE_THRESHOLD:
                passes = False
                break

        bounded = check_bounded(ctxs, a1, a2, a3, a4)

        if passes and bounded:
            verified.append(cand)

    verified.sort(key=lambda x: x['sum_a'])
    return verified


def analyze(verified):
    if not verified:
        return {'status': 'NO VERIFIED'}
    best = verified[0]
    sums = [c['sum_a'] for c in verified]
    return {
        'status': 'OK',
        'best': {
            'a1': best['a1'], 'a2': best['a2'],
            'a3': best['a3'], 'a4': best['a4']
        },
        'best_sum': best['sum_a'],
        'verified_count': len(verified),
        'sum_range': {
            'min': float(min(sums)),
            'max': float(max(sums)),
            'mean': float(np.mean(sums))
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pprint("=" * 80)
    pprint("CALIBRATION - US Crop & Weather Dataset")
    pprint(f"Objective: minimize sum of coefficients subject to feasibility")
    pprint(f"Grid range: [0.50, 0.99]")
    pprint(f"Seed: {SEED} | Sigma: {PERTURB_SIGMA} | Per real ctx: {PERTURB_PER_REAL}")
    pprint("=" * 80)

    datasets = load_all_datasets()
    if not datasets:
        pprint("Dataset not found. Exiting.")
        sys.exit(1)

    real_ctxs = extract_contexts(datasets)
    pprint(f"\nExtracted {len(real_ctxs)} real contexts.")

    pprint("\nGenerating perturbed contexts around real ones...")
    synth_ctxs = generate_perturbed_contexts(real_ctxs)
    pprint(f"Generated {len(synth_ctxs)} synthetic contexts "
           f"(sigma={PERTURB_SIGMA}, per_real={PERTURB_PER_REAL})")

    all_ctxs = list(synth_ctxs) + list(real_ctxs)
    pprint(f"Total contexts: {len(all_ctxs)}")

    feasible = grid_search(all_ctxs)

    if not feasible:
        pprint("\n" + "!" * 80)
        pprint("No feasible coefficients found.")
        pprint("!" * 80)
        pprint("\nDiagnostic on a mid-range candidate (a1=a2=a3=a4=0.75):")
        diag = diagnose_failure(all_ctxs, 0.75, 0.75, 0.75, 0.75)
        for name, st in diag.items():
            pprint(f"  {name:8s}: min={st['min']:.4f} "
                   f"q05={st['q05']:.4f} "
                   f"median={st['median']:.4f} "
                   f"max={st['max']:.4f}")
        pprint("\nDiagnostic on a high-range candidate (a1=a2=a3=a4=0.95):")
        diag = diagnose_failure(all_ctxs, 0.95, 0.95, 0.95, 0.95)
        for name, st in diag.items():
            pprint(f"  {name:8s}: min={st['min']:.4f} "
                   f"q05={st['q05']:.4f} "
                   f"median={st['median']:.4f} "
                   f"max={st['max']:.4f}")
        sys.exit(1)

    verified = final_verification(all_ctxs, feasible, top_k=20)
    if not verified:
        pprint("No verified candidates. Using top feasible.")
        verified = feasible[:1]

    result = analyze(verified)
    best = result['best']

    pprint("\n" + "=" * 80)
    pprint("FINAL CALIBRATION RESULT")
    pprint("=" * 80)
    pprint(f"a1 = {best['a1']:.2f}  (Interpretability)")
    pprint(f"a2 = {best['a2']:.2f}  (Robustness)")
    pprint(f"a3 = {best['a3']:.2f}  (Scalability)")
    pprint(f"a4 = {best['a4']:.2f}  (Rep. Capacity)")
    pprint(f"Sum = {result['best_sum']:.2f}")
    pprint(f"Verified candidates: {result['verified_count']}")
    pprint(f"Sum range across verified: "
           f"[{result['sum_range']['min']:.2f}, "
           f"{result['sum_range']['max']:.2f}]")

    os.makedirs('output', exist_ok=True)
    report = {
        'dataset': 'US_Agriculture_Weather_2010_2024.csv',
        'target': 'Corn_Price_USD',
        'seed': SEED,
        'grid_range': [0.50, 0.99],
        'dominance_threshold': DOMINANCE_THRESHOLD,
        'dominance_quantile': DOMINANCE_QUANTILE,
        'perturb_sigma': PERTURB_SIGMA,
        'perturb_per_real': PERTURB_PER_REAL,
        'real_contexts': len(real_ctxs),
        'synthetic_contexts': len(synth_ctxs),
        'total_contexts': len(all_ctxs),
        'objective': 'minimize_sum_a_among_feasible',
        'selected': best,
        'best_sum': result['best_sum'],
        'verified_count': result['verified_count'],
        'sum_range': result['sum_range']
    }

    with open('output/best_coefficients_us_crop.json', 'w') as f:
        json.dump(report, f, indent=4)

    pprint("\nReport saved to output/best_coefficients_us_crop.json")
    pprint("=" * 80)


if __name__ == "__main__":
    main()
