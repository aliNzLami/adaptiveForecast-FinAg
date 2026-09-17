import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Calibrated coefficients (from calibrate.py)
# ---------------------------------------------------------------------------
A1 = 0.98   # Interpretability
A2 = 0.95   # Robustness
A3 = 0.90   # Scalability
A4 = 0.85   # Representation Capacity

E_FIXED = 0.2   # Expertise level fixed for this study


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def compute_context_from_window(row):
    V = np.log10(90) / 6
    N = (0.3 * (row["corn_std"] / 50)
         + 0.3 * (row["volatility"] / 0.05)
         + 0.4 * (1 - abs(row["corr_price_temp"])))
    N = min(1.0, max(0.0, N))
    G = 0.5
    rho = 15 / 90
    return np.array([V, N, G, rho, E_FIXED])


def compute_requirements(ctx, a1, a2, a3, a4):
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
        "Hidden Markov Model":  np.array([0.85, 0.55, 0.30, 0.55]),
        "KNeighborsTimeSeries": np.array([0.70, 0.35, 0.50, 0.55]),
    }


def get_requirement_weights():
    w = np.array([A1, A2, A3, A4])
    return w / w.sum()


def manhattan_score(req, cap, weights):
    return 1.0 - np.sum(weights * np.abs(req - cap))


def get_model_instance(model_name):
    model_name = model_name.strip()
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "KNeighborsTimeSeries":
        return KNeighborsRegressor(n_neighbors=5)
    elif model_name == "XGBoost":
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    elif model_name == "LightGBM":
        return LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    elif model_name == "Hidden Markov Model":
        return DummyRegressor(strategy="mean")
    else:
        return None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    windows_path = os.path.join(project_root, "dataset", "window_features.csv")
    main_path = os.path.join(project_root, "dataset", "US_Agriculture_Weather_2010_2024.csv")
    ranking_output = os.path.join(project_root, "model_rankings.csv")
    performance_output = os.path.join(project_root, "top_models_performance.csv")

    print(f"Project root: {project_root}")
    print(f"Windows path: {windows_path}")
    print(f"Main data path: {main_path}")

    if not os.path.exists(windows_path):
        print("Error: window_features.csv not found.")
        sys.exit(1)

    # ============= PART 1: GENERATE RANKINGS =============
    df_windows = pd.read_csv(windows_path)
    df_windows["window_start"] = pd.to_datetime(df_windows["window_start"])
    df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])

    model_profiles = get_model_profiles()
    req_weights = get_requirement_weights()

    print(f"\nCoefficients: a1={A1}, a2={A2}, a3={A3}, a4={A4}")
    print(f"E (fixed): {E_FIXED}")
    print(f"Weights (normalised): {req_weights.round(4)}")

    ranking_results = []
    for idx, row in df_windows.iterrows():
        ctx = compute_context_from_window(row)
        req = compute_requirements(ctx, A1, A2, A3, A4)

        score_dict = {
            name: manhattan_score(req, cap, req_weights)
            for name, cap in model_profiles.items()
        }

        sorted_models = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        best_model, best_score = sorted_models[0]
        second_model, second_score = sorted_models[1]

        ranking_results.append({
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "best_model": best_model,
            "best_score": round(best_score, 4),
            "second_model": second_model,
            "second_score": round(second_score, 4),
            "score_gap": round(best_score - second_score, 4),
        })

    df_rankings = pd.DataFrame(ranking_results)
    df_rankings.to_csv(ranking_output, index=False)
    print(f"\nmodel_rankings.csv generated at {ranking_output}")

    print("\nBest model distribution:")
    print(df_rankings["best_model"].value_counts().to_string())

    print("\nSecond model distribution:")
    print(df_rankings["second_model"].value_counts().to_string())

    # ============= PART 2: EVALUATE PERFORMANCE =============
    if not os.path.exists(main_path):
        print("Error: Main dataset not found. Skipping performance evaluation.")
        return

    df_main = pd.read_csv(main_path)
    df_main["Date"] = pd.to_datetime(df_main["Date"])
    df_main = df_main.sort_values("Date").reset_index(drop=True)

    price_dict = dict(zip(df_main["Date"], df_main["Corn_Price_USD"]))
    features = ["Max_Temp_C", "Min_Temp_C", "Precipitation_mm"]

    performance_results = []

    for idx, row in df_rankings.iterrows():
        start = row["window_start"]
        end = row["window_end"]
        future_end = end + pd.Timedelta(days=30)

        if future_end not in price_dict:
            continue

        train_data = df_main[(df_main["Date"] >= start) & (df_main["Date"] <= end)]
        test_data = df_main[(df_main["Date"] > end) & (df_main["Date"] <= future_end)]

        if len(train_data) < 10 or len(test_data) < 3:
            continue

        X_train = train_data[features].values
        y_train = train_data["Corn_Price_USD"].values
        X_test = test_data[features].values
        y_test = test_data["Corn_Price_USD"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        result_row = {
            "window_start": start,
            "window_end": end,
            "best_model": row["best_model"],
            "second_model": row["second_model"],
        }

        best_model = get_model_instance(row["best_model"])
        if best_model is not None:
            best_model.fit(X_train_scaled, y_train)
            y_pred_best = best_model.predict(X_test_scaled)
            result_row["best_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_best))
            result_row["best_mae"] = mean_absolute_error(y_test, y_pred_best)
        else:
            result_row["best_rmse"] = None
            result_row["best_mae"] = None

        second_model = get_model_instance(row["second_model"])
        if second_model is not None:
            second_model.fit(X_train_scaled, y_train)
            y_pred_second = second_model.predict(X_test_scaled)
            result_row["second_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_second))
            result_row["second_mae"] = mean_absolute_error(y_test, y_pred_second)
        else:
            result_row["second_rmse"] = None
            result_row["second_mae"] = None

        if result_row["best_rmse"] is not None and result_row["second_rmse"] is not None:
            result_row["rmse_gap"] = result_row["best_rmse"] - result_row["second_rmse"]
        else:
            result_row["rmse_gap"] = None

        performance_results.append(result_row)

    if performance_results:
        df_performance = pd.DataFrame(performance_results)
        df_performance.to_csv(performance_output, index=False)
        print(f"\ntop_models_performance.csv generated at {performance_output}")

        valid = df_performance.dropna(subset=["best_rmse", "second_rmse"])
        if len(valid) > 0:
            print(f"\nAverage RMSE - Best Model: {valid['best_rmse'].mean():.4f}")
            print(f"Average RMSE - Second Model: {valid['second_rmse'].mean():.4f}")
            print(f"Average MAE  - Best Model: {valid['best_mae'].mean():.4f}")
            print(f"Average MAE  - Second Model: {valid['second_mae'].mean():.4f}")
            best_wins = (valid['best_rmse'] < valid['second_rmse']).sum()
            second_wins = (valid['second_rmse'] < valid['best_rmse']).sum()
            print(f"Best model wins in {best_wins} windows ({best_wins/len(valid)*100:.1f}%)")
            print(f"Second model wins in {second_wins} windows ({second_wins/len(valid)*100:.1f}%)")
            print(f"Total windows evaluated: {len(valid)}")
    else:
        print("No performance results generated.")


if __name__ == "__main__":
    main()
