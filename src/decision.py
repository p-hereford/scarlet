import pandas as pd


def classify_regime(row):
    z = row["fraud_zscore"]
    level = row["fraud_rolling_4w"]

    # ignore noise
    if level < 40:
        return "normal"

    # stressed: real spike + meaningful level
    if z > 1.25 and level > 100:
        return "stressed"

    # elevated: moderate abnormality (early warning)
    if 0.5 < z <= 1.25:
        return "elevated"

    return "normal"


def classify_direction(row):
    g = row["fraud_growth"]

    if pd.isna(g):
        return "stable"

    if g > 0.5:
        return "increasing"
    elif g < -0.5:
        return "decreasing"
    else:
        return "stable"


def build_decision(df):
    df = df.copy()

    df["regime"] = df.apply(classify_regime, axis=1)
    df["direction"] = df.apply(classify_direction, axis=1)

    decisions = []

    for _, row in df.iterrows():
        if row["regime"] == "stressed" and row["direction"] == "increasing":
            action = "tighten_controls"
        elif row["regime"] == "stressed":
            action = "maintain_high_alert"
        elif row["regime"] == "elevated" and row["direction"] == "increasing":
            action = "preemptive_tightening"
        elif row["regime"] == "elevated":
            action = "heightened_monitoring"
        elif row["regime"] == "normal" and row["direction"] == "decreasing":
            action = "relax_controls"
        else:
            action = "monitor"

        decisions.append(action)

    df["action"] = decisions

    return df