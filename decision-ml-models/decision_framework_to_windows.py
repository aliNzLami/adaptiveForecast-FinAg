import pandas as pd
import numpy as np
import os
import json


# ---------------------------------------------------------------------------
# Calibrated coefficients for this dataset
# ---------------------------------------------------------------------------
# Selected from the feasible region of the calibration grid. Sum = 3.68.
A1 = 0.98   # Interpretability
A2 = 0.95   # Robustness
A3 = 0.90   # Scalability
A4 = 0.85   # Representation Capacity

# Fixed expertise level. Low value because the target users are
# agricultural decision-makers without domain expertise in ML.
E_FIXED = 0.2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def compute_context_from_window(row):
    """
    Build the 5-dim context vector for a 90-day window.

    V   : data volume. 90 daily observations -> log10(90)/6.
    N   : noise level. Derived from price std, volatility,
          and price-temperature correlation.
    G   : granularity. Daily data -> 0.5 (medium-high resolution).
    rho : feature-to-instance ratio. 15 features / 90 days.
    E   : expertise level. Fixed at 0.2 for this study.
    """
    V = np.log10(90) / 6
    N = (0.3 * (row["corn_std"] / 50)
         + 0.3 * (row["volatility"] / 0.05)
         + 0.4 * (1 - abs(row["corr_price_temp"])))
    N = min(1.0, max(0.0, N))
    G = 0.5
    rho = 15 / 90
    return np.array([V, N, G, rho, E_FIXED])


def compute_requirements(ctx, a1, a2, a3, a4):
    """Requirement vector from context vector and calibrated coefficients."""
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal   = a3 * tanh(3 * V) + b3 * G
    r_rep    = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])


def get_model_profiles():
    return {
        "Random Forest":        np.array([0.65, 0.85, 0.65, 0.50]),
        "XGBoost":              np.array([0.50, 0.85, 0.75, 0.80]),
        "LightGBM":             np.array([0.50, 0.85, 0.90, 0.80]),
        "Hidden Markov Model":  np.array([0.80, 0.55, 0.30, 0.55]),
        "KNeighborsTimeSeries": np.array([0.75, 0.35, 0.50, 0.55]),
    }


def get_requirement_weights():
    """
    Normalised weights for the Manhattan compatibility function.
    Derived from the calibrated coefficients: each dimension's weight is its
    calibrated coefficient divided by the sum of all coefficients.
    """
    w = np.array([A1, A2, A3, A4])
    return w / w.sum()


def manhattan_score(req, cap, weights):
    return 1.0 - np.sum(weights * np.abs(req - cap))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "dataset", "window_features.csv")
    output_path = os.path.join(base_dir, "..", "model_recommendations.csv")

    df_windows = pd.read_csv(input_path)
    df_windows["window_start"] = pd.to_datetime(df_windows["window_start"])
    df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])

    model_profiles = get_model_profiles()
    req_weights = get_requirement_weights()

    print(f"Coefficients: a1={A1}, a2={A2}, a3={A3}, a4={A4}")
    print(f"E (fixed): {E_FIXED}")
    print(f"Weights (normalised): {req_weights.round(4)}")
    print()

    recommendations = []
    scores = []
    requirement_vectors = []

    for idx, row in df_windows.iterrows():
        ctx = compute_context_from_window(row)
        req = compute_requirements(ctx, A1, A2, A3, A4)

        score_dict = {
            name: manhattan_score(req, cap, req_weights)
            for name, cap in model_profiles.items()
        }

        best_model = max(score_dict, key=score_dict.get)
        best_score = score_dict[best_model]

        recommendations.append(best_model)
        scores.append(round(best_score, 4))
        requirement_vectors.append(req)

    df_windows["recommended_model"] = recommendations
    df_windows["compatibility_score"] = scores

    reqs = np.array(requirement_vectors)
    print("Requirement vector (mean over windows):")
    for i, name in enumerate(["interp", "robust", "scal", "rep"]):
        print(f"  {name:8s}: mean={reqs[:, i].mean():.4f}  "
              f"std={reqs[:, i].std():.4f}")

    df_windows.to_csv(output_path, index=False)
    print(f"\nmodel_recommendations.csv generated at: {output_path}")

    # Distribution of recommended models
    print("\nRecommended model distribution:")
    print(df_windows["recommended_model"].value_counts().to_string())


if __name__ == "__main__":
    main()
