import json
import numpy as np

# --- معادلات (همان) ---
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
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1-a1, 1-a2, 1-a3, 1-a4
    return np.array([
        a1 * (1 - sigmoid(10*(E-0.5))) + b1 * rho,
        a2 * sigmoid(12*(N-0.35)) + b2 * tanh(2*rho),
        a3 * tanh(3*V) + b3 * G,
        a4 * G + b4 * E
    ])

def compute_sens(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1-a1, 1-a2, 1-a3, 1-a4
    s = np.zeros((4,5))
    s[0,4] = -a1 * 10 * sigmoid_deriv(10*(E-0.5))
    s[0,3] = b1
    s[1,1] = a2 * 12 * sigmoid_deriv(12*(N-0.35))
    s[1,3] = b2 * 2 * tanh_deriv(2*rho)
    s[2,0] = a3 * 3 * tanh_deriv(3*V)
    s[2,2] = b3
    s[3,2] = a4
    s[3,4] = b4
    return s

def dominance(ctx, a1, a2, a3, a4):
    pairs = [(0,4,3),(1,1,3),(2,0,2),(3,2,4)]
    sm = compute_sens(ctx, a1, a2, a3, a4)
    return np.array([
        abs(sm[r,p]) / (abs(sm[r,p]) + abs(sm[r,s]) + 1e-12)
        for r,p,s in pairs
    ])

# --- بازسازی زمینه‌ها ---
import pandas as pd
import os

def ctx_from_df(df, E):
    X = df.drop(columns=['Corn_Price_USD'], errors='ignore')
    n, p = X.shape
    V = np.clip(np.log10(max(n,1))/6, 0, 1)
    rho = np.clip(p/max(n,1), 0, 1)
    missing = X.isnull().sum().sum()/(n*p) if n*p>0 else 0
    out = 0
    for col in X.select_dtypes(include=[np.number]).columns:
        std = X[col].std()
        if std > 0:
            out += ((X[col]-X[col].mean()).abs() > 3*std).sum()/max(n,1)
    out = out / max(1, len(X.select_dtypes(include=[np.number]).columns))
    N = np.clip(0.5*missing + 0.5*out, 0, 1)
    dates = X.select_dtypes(include=['datetime64']).columns
    if len(dates) > 0:
        d = X[dates[0]].dropna().sort_values()
        if len(d) > 1:
            deltas = d.diff().dropna()
            md = deltas.median().total_seconds()
            G = np.clip(86400/max(md,86400), 0, 1)
        else:
            G = 0.5
    else:
        G = np.clip(np.log10(max(n,1))/6, 0, 1)
    return np.array([V, N, G, rho, E])

df = pd.read_csv('dataset/US_Agriculture_Weather_2010_2024.csv')
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date').reset_index(drop=True)

real_ctxs = [ctx_from_df(df, E) for E in [0.0, 0.25, 0.5, 0.75, 1.0]]

rng = np.random.default_rng(42)
synth = []
for c in real_ctxs:
    for _ in range(100):
        synth.append(np.clip(c + rng.normal(0, 0.05, 5), 0, 1))
all_ctxs = list(synth) + real_ctxs

# --- تست ضرایب دستی ---
a1, a2, a3, a4 = 0.98, 0.95, 0.90, 0.85

print(f"Testing coefficients: a1={a1}, a2={a2}, a3={a3}, a4={a4}")
print(f"Sum = {a1+a2+a3+a4:.4f}")
print()

# Boundedness
bounded_ok = True
for c in all_ctxs:
    r = compute_requirements(c, a1, a2, a3, a4)
    if np.any(r < 0) or np.any(r > 1):
        bounded_ok = False
        break
print(f"Boundedness: {'PASS' if bounded_ok else 'FAIL'}")

# Dominance at q05
all_D = np.array([dominance(c, a1, a2, a3, a4) for c in all_ctxs])
names = ['interp', 'robust', 'scal', 'rep']
print(f"\nDominance at q05 (threshold=0.70):")
for i, n in enumerate(names):
    q05 = np.quantile(all_D[:, i], 0.05)
    status = 'PASS' if q05 > 0.70 else 'FAIL'
    print(f"  {n:8s}: q05={q05:.4f}  median={np.median(all_D[:,i]):.4f}  [{status}]")

# Requirement vector
reqs = np.array([compute_requirements(c, a1, a2, a3, a4) for c in all_ctxs])
print(f"\nRequirement vector (mean over contexts):")
for i, n in enumerate(names):
    print(f"  {n:8s}: {reqs[:,i].mean():.4f}")
