import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # final_model_per_window.csv lives at the project root.
    input_path = os.path.join(base_dir, "dataset", "final_model_per_window.csv")
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

    transitions = []
    previous_model = df["final_model"].iloc[0]

    for idx in range(1, len(df)):
        current_model = df["final_model"].iloc[idx]

        if current_model == previous_model:
            continue

        # The new evidence for this transition lives in the 30-day slice
        # unique to the new window: from window_start[i] to
        # window_start[i] + 15 days (midpoint).
        new_slice_start = df["window_start"].iloc[idx]
        transition_estimate = new_slice_start + pd.Timedelta(days=15)

        transitions.append({
            "transition_index": idx,
            "window_start": new_slice_start,
            "window_end": df["window_end"].iloc[idx],
            "window_center": df["window_center"].iloc[idx],
            "transition_estimate": transition_estimate,
            "from_model": previous_model,
            "to_model": current_model,
        })
        previous_model = current_model

    if not transitions:
        print("No transitions found.")
        return

    df_transitions = pd.DataFrame(transitions)

    # Persistence: number of consecutive windows the new model survives
    # after the transition. Protects against spurious transitions caused
    # by the 60-day overlap between consecutive windows.
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
    df_transitions = df_transitions.drop(columns=["transition_index"])

    df_transitions.to_csv(output_path, index=False)

    print(f"Transitions saved to {output_path}")
    print(f"Number of transitions: {len(df_transitions)}")
    print()
    print("Persistence distribution (in windows):")
    print(df_transitions["persistence_windows"].describe().to_string())
    print()
    print("Short-lived transitions (persistence <= 1 window):")
    short = df_transitions[df_transitions["persistence_windows"] <= 1]
    print(f"  Count: {len(short)} out of {len(df_transitions)}")
    if len(short) > 0:
        print(short[["transition_estimate", "from_model", "to_model"]].to_string())
    print()
    print("Full transition list:")
    print(df_transitions[
        ["transition_estimate", "from_model", "to_model", "persistence_windows"]
    ].to_string())


if __name__ == "__main__":
    main()
