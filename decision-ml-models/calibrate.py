#!/usr/bin/env python3
import os
import sys
import json
import time
import warnings
from itertools import product

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
np.random.seed(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x)**2


def compute_requirements(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal = a3 * tanh(3 * V) + b3 * G
    r_rep = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])


def compute_analytical_sensitivity(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    sens = np.zeros((4, 5))
    sens[0, 4] = -a1 * 10 * sigmoid_deriv(10 * (E - 0.5))
    sens[0, 3] = b1
    sens[1, 1] = a2 * 12 * sigmoid_deriv(12 * (N - 0.35))
    sens[1, 3] = b2 * 2 * tanh_deriv(2 * rho)
    sens[2, 0] = a3 * 3 * tanh_deriv(3 * V)
    sens[2, 2] = b3
    sens[3, 2] = a4
    sens[3, 4] = b4
    return sens


def compute_dominance_ratio(ctx, a1, a2, a3, a4):
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sm = compute_analytical_sensitivity(ctx, a1, a2, a3, a4)
    for idx, (req, prim, sec) in enumerate(pairs):
        sp = abs(sm[req, prim])
        ss = abs(sm[req, sec])
        D[idx] = sp / (sp + ss + 1e-12)
    return D


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
        except:
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


def generate_perturbed_contexts(real_ctxs, n_per_real=100, sigma=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    synth = []
    for ctx in real_ctxs:
        for _ in range(n_per_real):
            noise = rng.normal(0, sigma, size=5)
            perturbed = np.clip(ctx + noise, 0, 1)
            synth.append(perturbed)
    return np.array(synth)


def check_dominance_quantile(ctxs, a1, a2, a3, a4, thresh=0.70, quantile=0.95, max_sample=2000):
    if len(ctxs) > max_sample:
        step = max(1, len(ctxs) // max_sample)
        sample_ctxs = [ctxs[i] for i in range(0, len(ctxs), step)][:max_sample]
    else:
        sample_ctxs = ctxs
    
    all_D = []
    for ctx in sample_ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        all_D.append(D)
    all_D = np.array(all_D)
    for req_idx in range(4):
        D_req = all_D[:, req_idx]
        q = np.quantile(D_req, quantile)
        if q <= thresh:
            return False
    return True


def check_bounded(ctxs, a1, a2, a3, a4):
    for ctx in ctxs:
        r = compute_requirements(ctx, a1, a2, a3, a4)
        if np.any(r < 0) or np.any(r > 1):
            return False
    return True


def compute_requirement_spread(ctxs, a1, a2, a3, a4):
    reqs = np.array([compute_requirements(ctx, a1, a2, a3, a4) for ctx in ctxs])
    stds = np.std(reqs, axis=0)
    return float(np.mean(stds))


def grid_search(ctxs, step=0.05):
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    pprint(f"\nGrid search over {total} combinations...")
    pprint(f"Threshold = 0.70 | Sampling {min(2000, len(ctxs))} contexts")
    
    start_time = time.time()
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        pprint(f"  {count}/{total} | {elapsed:.1f}s")
                    
                    if not check_dominance_quantile(ctxs, a1, a2, a3, a4):
                        continue
                    if not check_bounded(ctxs, a1, a2, a3, a4):
                        continue
                    
                    spread = compute_requirement_spread(ctxs, a1, a2, a3, a4)
                    feasible.append({
                        'a1': round(a1, 2), 'a2': round(a2, 2),
                        'a3': round(a3, 2), 'a4': round(a4, 2),
                        'sum_a': round(a1 + a2 + a3 + a4, 4),
                        'spread': round(spread, 6)
                    })
    
    feasible.sort(key=lambda x: x['spread'], reverse=True)
    pprint(f"Found {len(feasible)} feasible candidates")
    return feasible


def final_verification(ctxs, candidates, top_k=20):
    verified = []
    pprint(f"\nFinal verification on all {len(ctxs)} contexts (top {top_k})...")
    
    for cand in candidates[:top_k]:
        a1, a2, a3, a4 = cand['a1'], cand['a2'], cand['a3'], cand['a4']
        all_D = []
        for ctx in ctxs:
            D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
            all_D.append(D)
        all_D = np.array(all_D)
        
        passes = True
        for req_idx in range(4):
            D_req = all_D[:, req_idx]
            q = np.quantile(D_req, 0.95)
            if q <= 0.70:
                passes = False
                break
        
        bounded = True
        for ctx in ctxs:
            r = compute_requirements(ctx, a1, a2, a3, a4)
            if np.any(r < 0) or np.any(r > 1):
                bounded = False
                break
        
        if passes and bounded:
            verified.append(cand)
    
    verified.sort(key=lambda x: x['spread'], reverse=True)
    return verified


def analyze(verified):
    if not verified:
        return {'status': 'NO VERIFIED'}
    best = verified[0]
    spreads = [c['spread'] for c in verified]
    return {
        'status': 'OK',
        'best': {
            'a1': best['a1'], 'a2': best['a2'],
            'a3': best['a3'], 'a4': best['a4']
        },
        'best_sum': best['sum_a'],
        'best_spread': best['spread'],
        'verified_count': len(verified),
        'spread_range': {
            'min': float(min(spreads)),
            'max': float(max(spreads)),
            'mean': float(np.mean(spreads))
        }
    }


def main():
    pprint("=" * 80)
    pprint("CALIBRATION - US Crop & Weather Dataset")
    pprint(f"Target: Corn_Price_USD | Threshold: 0.70 | Seed: {SEED}")
    pprint("=" * 80)
    
    datasets = load_all_datasets()
    if not datasets:
        pprint("Dataset not found. Exiting.")
        sys.exit(1)
    
    real_ctxs = extract_contexts(datasets)
    pprint(f"\nExtracted {len(real_ctxs)} real contexts.")
    
    pprint("\nGenerating perturbed contexts around real ones...")
    synth_ctxs = generate_perturbed_contexts(real_ctxs, n_per_real=100, sigma=0.05)
    pprint(f"Generated {len(synth_ctxs)} synthetic contexts (sigma=0.05)")
    
    all_ctxs = list(synth_ctxs) + list(real_ctxs)
    pprint(f"Total contexts: {len(all_ctxs)}")
    
    feasible = grid_search(all_ctxs, step=0.05)
    
    if not feasible:
        pprint("No feasible coefficients found.")
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
    pprint(f"Spread (objective) = {result['best_spread']:.6f}")
    pprint(f"Verified candidates: {result['verified_count']}")
    pprint(f"Spread range: [{result['spread_range']['min']:.6f}, {result['spread_range']['max']:.6f}]")
    
    os.makedirs('output', exist_ok=True)
    report = {
        'dataset': 'US_Agriculture_Weather_2010_2024.csv',
        'target': 'Corn_Price_USD',
        'seed': SEED,
        'real_contexts': len(real_ctxs),
        'synthetic_contexts': len(synth_ctxs),
        'total_contexts': len(all_ctxs),
        'objective': 'maximize_requirement_spread',
        'selected': best,
        'best_sum': result['best_sum'],
        'best_spread': result['best_spread'],
        'verified_count': result['verified_count'],
        'spread_range': result['spread_range']
    }
    
    with open('output/best_coefficients_us_crop.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    pprint("\nReport saved to output/best_coefficients_us_crop.json")
    pprint("=" * 80)


if __name__ == "__main__":
    main()
