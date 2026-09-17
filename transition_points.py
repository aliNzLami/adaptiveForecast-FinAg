import os
import pandas as pd


# Confidence thresholds
GAP_HIGH = 5.0       # RMSE gap (absolute) for high-confidence transitions
GAP_MEDIUM = 2.0     # RMSE gap for medium-confidence transitions
PERSIST_HIGH = 2     # persistence in windows for high/medium confidence


def classify_confidence(rmse_gap, persistence_windows):
    """
    Classify a transition by confidence.

    high   : large RMSE gap AND the new model survives at least 2 windows
    medium : moderate RMSE gap OR the new model survives at least 2 windows
    low    : everything else (likely windowing noise)
    """
    gap = abs(rmse_gap)

    if gap >= GAP_HIGH and persistence_windows >= PERSIST_HIGH:
        return "high"
    if gap >= GAP_MEDIUM or persistence_windows >= PERSIST_HIGH:
        return "medium"
    return "low"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(base_dir, "final_model_per_window.csv")
    output_path = os.path.join(base_dir, "transition_points.csv")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    df = pd.read_csv(input_path)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])

    # window_center is not propagated through model_ranking.py and
    # final_model_per_window.py. Compute it if missing.
    if "window_center" in df.columns:
        df["window_center"] = pd.to_datetime(df["window_center"])
    else:
        df["window_center"] = df["window_start"] + (df["window_end"] - df["window_start"]) / 2

    df = df.sort_values("window_start").reset_index(drop=True)

    # Filter out Unknown models. They are not valid regime labels.
    n_unknown = (df["final_model"] == "Unknown").sum()
    if n_unknown > 0:
        print(f"Warning: {n_unknown} windows have final_model='Unknown'. "
              f"Excluded from transition detection.")
        df = df[df["final_model"] != "Unknown"].reset_index(drop=True)

    if len(df) < 2:
        print("Not enough windows to detect transitions.")
        return

    # Detect transitions.
    transitions = []
    previous_model = df["final_model"].iloc[0]

    for idx in range(1, len(df)):
        current_model = df["final_model"].iloc[idx]

        if current_model == previous_model:
            continue

        new_slice_start = df["window_start"].iloc[idx]
        transition_estimate = new_slice_start + pd.Timedelta(days=15)

        # RMSE gap at the transition window: |best_rmse - second_rmse|
        gap_at_transition = abs(
            df["best_rmse"].iloc[idx] - df["second_rmse"].iloc[idx]
        )

        transitions.append({
            "transition_index": idx,
            "window_start": new_slice_start,
            "window_end": df["window_end"].iloc[idx],
            "window_center": df["window_center"].iloc[idx],
            "transition_estimate": transition_estimate,
            "from_model": previous_model,
            "to_model": current_model,
            "rmse_gap_at_transition": gap_at_transition,
        })
        previous_model = current_model

    if not transitions:
        print("No transitions found.")
        return

    df_transitions = pd.DataFrame(transitions)

    # Persistence: consecutive windows the new model survives after the transition.
    persistence_windows = []
    persistence_days = []
    for _, tr in df_transitions.iterrows():
        idx = int(tr["transition_index"])
        new_model = tr["to_model"]
        count = 1
        j = idx + 1
        while j < len(df) and df["final_model"].iloc[j] == new_model:
            count += 1
            j += 1
        persistence_windows.append(count)
        if j < len(df):
            days = (df["window_start"].iloc[j] - df["window_start"].iloc[idx]).days
        else:
            days = (df["window_end"].iloc[-1] - df["window_start"].iloc[idx]).days
        persistence_days.append(days)

    df_transitions["persistence_windows"] = persistence_windows
    df_transitions["persistence_days"] = persistence_days

    # Confidence classification.
    df_transitions["confidence"] = df_transitions.apply(
        lambda r: classify_confidence(r["rmse_gap_at_transition"],
                                       r["persistence_windows"]),
        axis=1
    )

    df_transitions = df_transitions.drop(columns=["transition_index"])

    # Order columns for readability.
    cols = [
        "transition_estimate", "window_start", "window_end", "window_center",
        "from_model", "to_model",
        "rmse_gap_at_transition",
        "persistence_windows", "persistence_days",
        "confidence",
    ]
    df_transitions = df_transitions[cols]

    df_transitions.to_csv(output_path, index=False)

    # Summary report.
    print(f"Transitions saved to {output_path}")
    print(f"Number of transitions: {len(df_transitions)}")
    print()

    print("Confidence distribution:")
    print(df_transitions["confidence"].value_counts().to_string())
    print()

    print("Persistence distribution (in windows):")
    print(df_transitions["persistence_windows"].describe().to_string())
    print()

    print("Transitions by confidence level:")
    for level in ["high", "medium", "low"]:
        sub = df_transitions[df_transitions["confidence"] == level]
        if len(sub) == 0:
            continue
        print(f"\n--- {level.upper()} ({len(sub)}) ---")
        print(sub[
            ["transition_estimate", "from_model", "to_model",
             "rmse_gap_at_transition", "persistence_windows"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
