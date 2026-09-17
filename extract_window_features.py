import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def load_and_clean_data(file_path):
    """
    Load and clean the raw dataset.

    Invalid date entries are coerced to NaT and dropped. This matches
    the paper's claim that invalid date entries were excluded during
    preprocessing. Using errors='coerce' avoids a crash on bad entries.
    """
    df = pd.read_csv(file_path)
    n_before = len(df)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')

    n_invalid = df["Date"].isna().sum()
    if n_invalid > 0:
        print(f"Dropped {n_invalid} rows with invalid or missing dates")
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
    else:
        print("No invalid dates found.")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _stats(series):
    """
    Mean, std, skewness, and standard kurtosis of a series.

    Uses fisher=False so that a normal distribution yields kurtosis = 3.0
    (standard definition), not 0.0 (excess kurtosis).
    """
    s = series.dropna()
    return {
        "mean":     s.mean(),
        "std":      s.std(),
        "skew":     skew(s) if len(s) > 2 else np.nan,
        "kurtosis": kurtosis(s, fisher=False) if len(s) > 3 else np.nan,
    }


def compute_window_features(df, window_days=90, step_days=30):
    """
    Sliding window feature extraction.

    Each window contains exactly `window_days` consecutive trading days.
    Daily returns use log differences, matching the paper's description
    of "daily logarithmic returns".

    Feature count:
        The paper's text says 15 features but its own Table 1 lists 11.
        We implement the 11 that are actually described:
        corn_mean, corn_std, corn_skew, corn_kurtosis,
        temp_mean, temp_std, temp_skew, temp_kurtosis,
        volatility, corr_price_temp, precip_mean.

    Window count:
        For n rows, window=90, step=30:
        n_windows = floor((n - 90) / 30) + 1.
        For n=3659 this yields 119 windows.
    """
    features_list = []
    n = len(df)

    for start_idx in range(0, n - window_days + 1, step_days):
        window_df = df.iloc[start_idx:start_idx + window_days].reset_index(drop=True)

        corn_stats = _stats(window_df["Corn_Price_USD"])

        temp_avg = (window_df["Max_Temp_C"] + window_df["Min_Temp_C"]) / 2
        temp_stats = _stats(temp_avg)

        # Log returns: log(P_t / P_{t-1})
        prices = window_df["Corn_Price_USD"]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        volatility = log_returns.std()

        corr_price_temp = window_df["Corn_Price_USD"].corr(temp_avg)

        features = {
            "window_start":  window_df["Date"].iloc[0],
            "window_end":    window_df["Date"].iloc[-1],
            "window_center": window_df["Date"].iloc[len(window_df) // 2],

            "corn_mean":     corn_stats["mean"],
            "corn_std":      corn_stats["std"],
            "corn_skew":     corn_stats["skew"],
            "corn_kurtosis": corn_stats["kurtosis"],

            "temp_mean":     temp_stats["mean"],
            "temp_std":      temp_stats["std"],
            "temp_skew":     temp_stats["skew"],
            "temp_kurtosis": temp_stats["kurtosis"],

            "volatility":       volatility,
            "corr_price_temp":  corr_price_temp,
            "precip_mean":      window_df["Precipitation_mm"].mean(),
        }
        features_list.append(features)

    return pd.DataFrame(features_list)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_file  = os.path.join(base_dir, "dataset", "US_Agriculture_Weather_2010_2024.csv")
    output_file = os.path.join(base_dir, "window_features.csv")

    df = load_and_clean_data(input_file)
    print(f"Cleaned rows: {len(df)}")

    features_df = compute_window_features(df)
    print(f"Windows extracted: {len(features_df)}")
    n_features = len([c for c in features_df.columns if not c.startswith("window_")])
    print(f"Features per window: {n_features}")

    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
