import pandas as pd
from decision import build_decision

df = pd.read_csv("data/jpm_weekly_features.csv")

df = build_decision(df)

print(df.tail(10)[["week", "fraud_rolling_4w", "fraud_growth", "regime", "direction", "action"]])