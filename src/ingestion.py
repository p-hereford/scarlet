"""
Fetches two complaint streams from the CFPB API and one news signal
from the GDELT API, then saves them to the data directory.

    data/jpm_complaints.csv      -- JPMorgan Chase complaints (unchanged)
    data/industry_complaints.csv -- all-bank fraud-related complaints
    data/gdelt_news.csv          -- weekly JPM fraud news article counts
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


"""
Paths
"""

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

JPM_OUTPUT = DATA_DIR / "jpm_complaints.csv"
INDUSTRY_OUTPUT = DATA_DIR / "industry_complaints.csv"
GDELT_OUTPUT = DATA_DIR / "gdelt_news.csv"

"""
Shared config
"""

CFPB_BASE_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

DATE_START = "2022-01-01"
DATE_END = "2024-12-31"

"""
The product list was kept tight because once one starts bringing in loosely related complaint 
categories the fraud signal gets diluted and stops being useful as an early warning.
"""
FRAUD_PRODUCTS = [
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
]


"""
Shared helpers
"""

def generate_month_ranges(start: str = DATE_START, end: str = DATE_END):
    """
    The pulls were split into monthly windows because the CFPB API was not truly reliable
    on large requests and it was better to have a slower pull than a broken one.
    """
    start_date = datetime.fromisoformat(start)
    end_date = datetime.fromisoformat(end)
    current = start_date
    while current < end_date:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current.strftime("%Y-%m-%d"), next_month.strftime("%Y-%m-%d")
        current = next_month


def _fetch_cfpb_page(params: dict, max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            response = requests.get(CFPB_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("hits", {}).get("hits", [])
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2)
    return []


def _fetch_cfpb_range(
    start_date: str,
    end_date: str,
    extra_params: dict,
    page_size: int = 100,
    max_pages: int = 50,
) -> list:
    all_hits = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "size": page_size,
            "from": offset,
            "date_received_min": start_date,
            "date_received_max": end_date,
            **extra_params,
        }
        hits = _fetch_cfpb_page(params)
        if not hits:
            break
        all_hits.extend(hits)
        offset += page_size
        time.sleep(0.1)

    return all_hits


def _process_cfpb_records(raw_data: list) -> pd.DataFrame:
    records = [
        {
            "date_received": item.get("_source", {}).get("date_received"),
            "product": item.get("_source", {}).get("product"),
            "issue": item.get("_source", {}).get("issue"),
            "state": item.get("_source", {}).get("state"),
            "company": item.get("_source", {}).get("company"),
            "narrative": item.get("_source", {}).get("consumer_complaint_narrative"),
        }
        for item in raw_data
    ]
    df = pd.DataFrame(records)
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received"])
    return df.sort_values("date_received").reset_index(drop=True)


"""
Source 1: JPMorgan Chase complaints
"""
def fetch_jpm_complaints() -> pd.DataFrame:
    print("\n[1/3] Fetching JPMorgan Chase complaints...")
    all_hits = []

    for start, end in generate_month_ranges():
        print(f"  JPM {start} -> {end}")
        hits = _fetch_cfpb_range(
            start_date=start,
            end_date=end,
            extra_params={"company": "JPMORGAN CHASE & CO."},
        )
        all_hits.extend(hits)

    df = _process_cfpb_records(all_hits)
    print(f"  JPM total records: {len(df)}")
    return df

"""
Source 2: Industry-wide fraud complaints
"""

def fetch_industry_complaints() -> pd.DataFrame:
    """
    The data was not filtered by company because the whole point is to pick up fraud pressure
    showing up across the market before it fully lands in JPM's own series.
    """
    print("\n[2/3] Fetching industry-wide fraud complaints...")
    all_hits = []

    for product in FRAUD_PRODUCTS:
        print(f"  Product: {product}")
        for start, end in generate_month_ranges():
            hits = _fetch_cfpb_range(
                start_date=start,
                end_date=end,
                extra_params={"product": product},
            )
            all_hits.extend(hits)
            time.sleep(0.05)

    df = _process_cfpb_records(all_hits)
    print(f"  Industry total records: {len(df)}")
    return df


"""
Source 3: GDELT news signal
"""

def _fetch_gdelt_week(start_date: str, end_date: str, query: str) -> int:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 250,
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
        "format": "json",
    }

    try:
        response = requests.get(GDELT_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return len(articles)
    except Exception:
        """
        Zero is returned on GDELT failure because it is a secondary signal and it would not have been 
        a good idea by any means to let a flaky external feed take the whole ingestion run down.
        """
        return 0


def fetch_gdelt_news() -> pd.DataFrame:

    """
    News was pulled in because complaints only show up once people actually file them,
    and media coverage can bring that forward by pushing latent victims to act.
    """

    print("\n[3/3] Fetching GDELT news signal...")

    queries = ["JPMorgan fraud", "Chase scam"]
    week_counts: dict = {}

    start_dt = datetime.fromisoformat(DATE_START)
    end_dt = datetime.fromisoformat(DATE_END)
    current = start_dt

    while current < end_dt:
        week_end = current + timedelta(days=6)
        if week_end > end_dt:
            week_end = end_dt

        start_str = current.strftime("%Y-%m-%d")
        end_str = week_end.strftime("%Y-%m-%d")
        week_key = current.strftime("%Y-%m-%d")

        total = 0
        for query in queries:
            count = _fetch_gdelt_week(start_str, end_str, query)
            total += count
            time.sleep(0.5)

        week_counts[week_key] = total
        print(f"  GDELT {start_str}: {total} articles")
        current += timedelta(days=7)

    df = pd.DataFrame(
        list(week_counts.items()),
        columns=["week", "news_article_count"],
    )
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)

    print(f"  GDELT weeks fetched: {len(df)}")
    return df


"""
Entry point
"""


def main() -> None:
    print("Starting SCARLET data ingestion...")
    print(f"Date range: {DATE_START} to {DATE_END}")

    jpm_df = fetch_jpm_complaints()
    jpm_df.to_csv(JPM_OUTPUT, index=False)
    print(f"  Saved -> {JPM_OUTPUT}")

    industry_df = fetch_industry_complaints()
    industry_df.to_csv(INDUSTRY_OUTPUT, index=False)
    print(f"  Saved -> {INDUSTRY_OUTPUT}")

    gdelt_df = fetch_gdelt_news()
    gdelt_df.to_csv(GDELT_OUTPUT, index=False)
    print(f"  Saved -> {GDELT_OUTPUT}")

    print("\nIngestion complete.")
    print(f"  JPM records      : {len(jpm_df)}")
    print(f"  Industry records : {len(industry_df)}")
    print(f"  GDELT weeks      : {len(gdelt_df)}")


if __name__ == "__main__":
    main()