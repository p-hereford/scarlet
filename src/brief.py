"""
This was kept separate because otherwise the brief ends up depending
on a layer that has not yet earned my full trust, unfortunately.
"""

import pandas as pd
from decision import build_decision


def build_brief(df):
    latest = df.iloc[-1]

    week = latest["week"]
    regime = latest["regime"].upper()
    direction = latest["direction"].upper()
    action = latest["action"].replace("_", " ").upper()

    fraud_complaints = latest["fraud_complaints"]
    rolling = latest["fraud_rolling_4w"]
    growth = latest["fraud_growth"]
    zscore = latest["fraud_zscore"]

    lines = []
    lines.append(f"SCARLET Risk Brief — Week of {week}")
    lines.append("")
    lines.append("Current state:")
    lines.append(f"- Regime: {regime}")
    lines.append(f"- Direction: {direction}")
    lines.append(f"- Recommended action: {action}")
    lines.append("")
    lines.append("Signal summary:")
    lines.append(f"- Fraud complaints this week: {fraud_complaints}")
    lines.append(f"- 4-week rolling average: {rolling:.1f}")
    lines.append(f"- Week-on-week fraud growth: {growth:.2f}")
    lines.append(f"- Fraud z-score: {zscore:.2f}")
    lines.append("")
    lines.append("Interpretation:")

    if latest["regime"] == "stressed" and latest["direction"] == "increasing":
        lines.append(
            "- Fraud pressure is materially above recent baseline and still accelerating. "
            "Control tightening is warranted to reduce loss exposure and downstream operational strain."
        )
    elif latest["regime"] == "elevated" and latest["direction"] == "increasing":
        lines.append(
            "- Fraud pressure is above normal levels and building. "
            "Pre-emptive tightening and closer monitoring are appropriate."
        )
    elif latest["regime"] == "elevated":
        lines.append(
            "- Fraud conditions are elevated relative to normal behaviour. "
            "Maintain heightened monitoring and prepare intervention if pressure worsens."
        )
    elif latest["direction"] == "decreasing":
        lines.append(
            "- Fraud pressure is easing. "
            "A more relaxed control posture may be justified if this trend persists."
        )
    else:
        lines.append(
            "- Fraud conditions appear broadly stable. "
            "Continue monitoring for further regime change."
        )

    return "\n".join(lines)


def main():
    df = pd.read_csv("data/jpm_weekly_features.csv")
    df = build_decision(df)
    brief = build_brief(df)
    print(brief)


if __name__ == "__main__":
    main()