import pandas as pd
from decision import build_decision


def evaluate_system(df):
    results = {}

    regime_counts = df["regime"].value_counts(normalize=True)
    action_counts = df["action"].value_counts(normalize=True)

    escalation_actions = [
        "heightened_monitoring",
        "preemptive_tightening",
        "tighten_controls"
    ]

    escalation_rate = df["action"].isin(escalation_actions).mean()
    stressed_rate = (df["regime"] == "stressed").mean()

    df = df.copy()

    df["future_spike"] = df["fraud_complaints"].shift(-1) > df["fraud_rolling_4w"]

    """
    Elevated and stressed were grouped together here because both are meant to function as 
    early warning states rather than only as descriptions of what is already obvious.
    """
    lead_signal = df[df["regime"].isin(["elevated", "stressed"])]
    lead_accuracy = lead_signal["future_spike"].mean()

    df["lag_warning"] = (
        (df["fraud_growth_lag1"] > 0.5) |
        (df["fraud_zscore_lag1"] > 0.75)
    )

    lag_warning_accuracy = df.loc[df["lag_warning"], "future_spike"].mean()

    results["regime_distribution"] = regime_counts
    results["action_distribution"] = action_counts
    results["escalation_rate"] = escalation_rate
    results["stressed_rate"] = stressed_rate
    results["lead_indicator_accuracy"] = lead_accuracy
    results["lag_warning_accuracy"] = lag_warning_accuracy

    return results


def print_evaluation(results):
    print("\nSCARLET Evaluation Summary\n")

    print("Regime distribution:")
    print(results["regime_distribution"])
    print()

    print("Action distribution:")
    print(results["action_distribution"])
    print()

    print(f"Escalation rate: {results['escalation_rate']:.2f}")
    print(f"Stressed rate: {results['stressed_rate']:.2f}")
    print(f"Lead indicator accuracy: {results['lead_indicator_accuracy']:.2f}")
    print(f"Lag warning accuracy: {results['lag_warning_accuracy']:.2f}")
    print()


def main():
    df = pd.read_csv("data/jpm_weekly_features.csv")
    df = build_decision(df)

    results = evaluate_system(df)
    print_evaluation(results)


if __name__ == "__main__":
    main()