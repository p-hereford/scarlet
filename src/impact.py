import pandas as pd
from pathlib import Path

from decision import build_decision


"""
This impact layer is only a rough translation from signal to money, so the numbers here are
scenario assumptions rather than realised economics.
"""
LOSS_PER_FRAUD_COMPLAINT = 250

"""
These reduction rates were set to scale with action intensity so the output behaves like a decision scenario,
 not a claim that controls reduce loss by a fixed observed amount.
"""
REDUCTION_RATES = {
    "monitor": 0.00,
    "relax_controls": 0.00,
    "heightened_monitoring": 0.05,
    "preemptive_tightening": 0.15,
    "tighten_controls": 0.25,
    "maintain_high_alert": 0.10,
}


def estimate_impact(df):
    df = df.copy()

    df["loss_per_fraud_complaint"] = LOSS_PER_FRAUD_COMPLAINT
    df["reduction_rate"] = df["action"].map(REDUCTION_RATES).fillna(0)

    df["estimated_exposure"] = (
        df["fraud_complaints"] * df["loss_per_fraud_complaint"]
    )

    df["estimated_loss_avoided"] = (
        df["estimated_exposure"] * df["reduction_rate"]
    )

    return df


def build_impact_summary(df):
    latest = df.iloc[-1]

    summary = [
        f"SCARLET Economic Impact — Week of {latest['week']}",
        "",
        "Current action:",
        f"- {latest['action'].upper()}",
        "",
        "Signal:",
        f"- Fraud complaints: {int(latest['fraud_complaints'])}",
        f"- Estimated exposure: ${latest['estimated_exposure']:,.0f}",
        "",
        "Impact estimate:",
        f"- Assumed reduction rate: {latest['reduction_rate']:.0%}",
        f"- Estimated loss avoided: ${latest['estimated_loss_avoided']:,.0f}",
        "",
        "Assumptions:",
        f"- Loss proxy per fraud complaint: ${LOSS_PER_FRAUD_COMPLAINT}",
        "- Reduction rate linked to action intensity",
        "- Scenario-based estimate (not realised P&L)",
    ]

    return "\n".join(summary)


def main():
    data_path = Path("data/jpm_weekly_features.csv")

    if not data_path.exists():
        print("Missing data file. Run feature_engineering first.")
        return

    df = pd.read_csv(data_path)
    df = build_decision(df)
    df = estimate_impact(df)

    print(df.tail(10)[[
        "week",
        "fraud_complaints",
        "action",
        "estimated_exposure",
        "estimated_loss_avoided"
    ]])

    print("\n")
    print(build_impact_summary(df))


if __name__ == "__main__":
    main()