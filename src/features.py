import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

JPM_INPUT = DATA_DIR / "jpm_complaints.csv"
INDUSTRY_INPUT = DATA_DIR / "industry_complaints.csv"
GDELT_INPUT = DATA_DIR / "gdelt_news.csv"
OUTPUT_FILE = DATA_DIR / "jpm_weekly_features.csv"


FRAUD_KEYWORDS = [
    "fraud",
    "scam",
    "unauthorized",
    "unauthorised",
    "identity theft",
    "stolen",
    "chargeback",
    "zelle",
    "wire transfer",
    "paypal",
    "venmo",
    "cash app",
]


def _to_week(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    df["week"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    return df


def _flag_fraud(df: pd.DataFrame, issue_col: str, narrative_col: str) -> pd.DataFrame:
    issue_text = df[issue_col].fillna("").str.lower()
    narrative_text = df[narrative_col].fillna("").str.lower()
    combined = issue_text + " " + narrative_text
    df["is_fraud"] = combined.apply(
        lambda text: any(kw in text for kw in FRAUD_KEYWORDS)
    )
    return df


def load_jpm(path: Path = JPM_INPUT) -> pd.DataFrame:
    print("Loading JPM complaints...")
    df = pd.read_csv(path)
    df = _to_week(df, "date_received")
    df = _flag_fraud(df, issue_col="issue", narrative_col="narrative")
    return df


def aggregate_jpm_weekly(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating JPM weekly...")
    weekly = (
        df.groupby("week")
        .agg(
            total_complaints=("issue", "count"),
            fraud_complaints=("is_fraud", "sum"),
        )
        .reset_index()
        .sort_values("week")
        .reset_index(drop=True)
    )
    return weekly


def load_industry(path: Path = INDUSTRY_INPUT) -> pd.DataFrame:
    print("Loading industry complaints...")
    df = pd.read_csv(path, low_memory=False)
    df = _to_week(df, "Date received")
    df = _flag_fraud(
        df,
        issue_col="Issue",
        narrative_col="Consumer complaint narrative",
    )
    return df


def aggregate_industry_weekly(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating industry weekly...")
    weekly = (
        df.groupby("week")
        .agg(industry_fraud_complaints=("is_fraud", "sum"))
        .reset_index()
        .sort_values("week")
        .reset_index(drop=True)
    )
    return weekly


def load_gdelt(path: Path = GDELT_INPUT) -> pd.DataFrame | None:
    """
    GDELT was left optional because it helps when it is there,but it was not
    worth making the whole feature table depend on it.
    """
    if not path.exists():
        print("GDELT file not found -- skipping news features.")
        return None
    print("Loading GDELT news...")
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["week"])
    return df[["week", "news_article_count"]].sort_values("week").reset_index(drop=True)


def compute_jpm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("week").copy()

    df["fraud_rate"] = df["fraud_complaints"] / df["total_complaints"]

    prev_fraud = df["fraud_complaints"].shift(1)
    df["fraud_growth"] = (df["fraud_complaints"] - prev_fraud) / prev_fraud
    """
    Growth was clipped because small prior values can create silly jumps 
    that look dramatic numerically and mean very little operationally.
    """
    df["fraud_growth"] = (
        df["fraud_growth"]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
        .clip(-3, 3)
    )

    df["fraud_rolling_4w"] = (
        df["fraud_complaints"].rolling(window=4, min_periods=4).mean()
    )
    df["fraud_std_4w"] = (
        df["fraud_complaints"].rolling(window=4, min_periods=4).std()
    )
    df["fraud_zscore"] = (
        (df["fraud_complaints"] - df["fraud_rolling_4w"]) / df["fraud_std_4w"]
    ).fillna(0).clip(-5, 5)

    df["fraud_complaints_lag1"] = df["fraud_complaints"].shift(1).fillna(0)
    df["fraud_growth_lag1"] = df["fraud_growth"].shift(1).fillna(0)
    df["fraud_zscore_lag1"] = df["fraud_zscore"].shift(1).fillna(0)
    df["fraud_rolling_4w_lag1"] = df["fraud_rolling_4w"].shift(1).fillna(0)

    """
    This duplicate name exists because the forecast file expects it exactly like this
     and changing it here would create avoidable friction across the pipeline.
    """
    df["fraud_complaints_roll_mean_4w_lag1"] = df["fraud_rolling_4w_lag1"]

    return df


def compute_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The industry features were all lagged because they are meant to act as an upstream warning signal,
     not to smuggle future information into the target week.
    """
    df = df.sort_values("week").copy()

    df["industry_fraud_complaints_lag1"] = (
        df["industry_fraud_complaints"].shift(1).fillna(0)
    )

    prev_industry = df["industry_fraud_complaints"].shift(1)
    industry_growth = (
        (df["industry_fraud_complaints"] - prev_industry) / prev_industry
    ).replace([float("inf"), -float("inf")], 0).fillna(0).clip(-3, 3)
    df["industry_fraud_growth_lag1"] = industry_growth.shift(1).fillna(0)

    df["industry_rolling_4w_lag1"] = (
        df["industry_fraud_complaints"]
        .rolling(window=4, min_periods=4)
        .mean()
        .shift(1)
        .fillna(0)
    )

    return df


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    industry = df["industry_fraud_complaints"].replace(0, float("nan"))
    df["jpm_industry_ratio"] = (df["fraud_complaints"] / industry).fillna(0)
    df["jpm_industry_ratio_lag1"] = df["jpm_industry_ratio"].shift(1).fillna(0)
    return df


def compute_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The news signal was only used in lagged form because same-week coverage would be too close
    to the target to treat cleanly as advance information.
    """
    df = df.copy()
    df["news_article_count_lag1"] = df["news_article_count"].shift(1).fillna(0)
    df["news_article_count_lag2"] = df["news_article_count"].shift(2).fillna(0)
    return df


def compute_weeks_since_nonzero(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    last_idx = None
    for i, v in enumerate(df["fraud_complaints"]):
        out.append(float("nan") if last_idx is None else float(i - last_idx))
        if v > 0:
            last_idx = i
    df = df.copy()
    """
    This was lagged as well because even a simple recency counter stops being clean
     the moment it is allowed to look at the current week.
    """
    df["weeks_since_nonzero"] = pd.Series(out, index=df.index).shift(1).fillna(0)
    return df


def build_feature_table() -> pd.DataFrame:
    jpm_weekly = aggregate_jpm_weekly(load_jpm())
    industry_weekly = aggregate_industry_weekly(load_industry())
    gdelt_weekly = load_gdelt()

    print("Merging sources...")
    df = jpm_weekly.merge(industry_weekly, on="week", how="left")

    if gdelt_weekly is not None:
        df = df.merge(gdelt_weekly, on="week", how="left")
        df["news_article_count"] = df["news_article_count"].fillna(0)

    df["industry_fraud_complaints"] = df["industry_fraud_complaints"].fillna(0)
    df = df.sort_values("week").reset_index(drop=True)

    print("Computing features...")
    df = compute_jpm_features(df)
    df = compute_industry_features(df)
    df = compute_ratio_features(df)
    df = compute_weeks_since_nonzero(df)

    if gdelt_weekly is not None:
        df = compute_news_features(df)

    return df


def main() -> None:
    df = build_feature_table()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")

    print("\nLeading indicator columns:")
    leading = [c for c in df.columns if "industry" in c or "news" in c or "ratio" in c]
    for col in leading:
        non_zero = (df[col] > 0).sum()
        print(f"  {col:45s} non-zero weeks: {non_zero}")


if __name__ == "__main__":
    main()