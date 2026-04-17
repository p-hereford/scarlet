import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "jpm_weekly_features.csv"


def main():
    print("Loading features...")
    df = pd.read_csv(INPUT_FILE)

    df["week"] = pd.to_datetime(df["week"])

    plt.figure()
    plt.title("Fraud Complaints Over Time")
    plt.plot(df["week"], df["fraud_complaints"])
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.figure()
    plt.title("Fraud Growth Rate")
    plt.plot(df["week"], df["fraud_growth"])
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.figure()
    plt.title("4 Week Rolling Average")
    plt.plot(df["week"], df["fraud_rolling_4w"])
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()