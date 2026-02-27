"""
compute_ath_proximity.py
Compute % of S&P 500 stocks within 3% of their all-time high (available data),
aggregated per GICS sector. Uses prices.parquet + universe_latest.parquet from sp500-breadth.

Output: data/breadth/breadth_sector_ath.csv
Columns: sector, near_ath_pct, total_tickers, count_near_ath, date
"""
from pathlib import Path
import pandas as pd

PRICES_PATH   = Path("/Users/erikgustafsson/.openclaw/workspace/projects/sp500-breadth/data/prices.parquet")
UNIVERSE_PATH = Path("/Users/erikgustafsson/.openclaw/workspace/projects/sp500-breadth/data/universe_latest.parquet")
OUT_PATH      = Path("/Users/erikgustafsson/.openclaw/workspace/projects/hub-cloud/data/breadth/breadth_sector_ath.csv")

def main():
    print("Loading prices...")
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])

    latest_date = prices["date"].max()
    print(f"Latest price date: {latest_date.date()}")

    # Latest close per ticker
    latest_px = (
        prices[prices["date"] == latest_date][["ticker", "Close"]]
        .copy()
        .rename(columns={"Close": "current_close"})
    )

    # ATH = max Close across all available history per ticker
    ath = (
        prices.groupby("ticker")["Close"]
        .max()
        .reset_index()
        .rename(columns={"Close": "ath"})
    )

    df = latest_px.merge(ath, on="ticker", how="inner")
    df["pct_from_ath"] = (df["ath"] - df["current_close"]) / df["ath"]
    df["within_3pct"] = df["pct_from_ath"] <= 0.03

    # Load sector mapping
    universe = pd.read_parquet(UNIVERSE_PATH)[["ticker", "sector"]]
    df = df.merge(universe, on="ticker", how="left")

    # Index-wide
    idx_row = {
        "sector":         "S&P 500 (Index)",
        "near_ath_pct":   round(df["within_3pct"].mean() * 100, 1),
        "total_tickers":  len(df),
        "count_near_ath": int(df["within_3pct"].sum()),
        "date":           str(latest_date.date()),
    }

    # Per sector
    sector_agg = (
        df.dropna(subset=["sector"])
        .groupby("sector")["within_3pct"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    sector_agg.columns = ["sector", "near_ath_frac", "count_near_ath", "total_tickers"]
    sector_agg["near_ath_pct"] = (sector_agg["near_ath_frac"] * 100).round(1)
    sector_agg["date"] = str(latest_date.date())
    sector_agg = sector_agg.drop(columns=["near_ath_frac"])

    result = pd.concat(
        [sector_agg, pd.DataFrame([idx_row])],
        ignore_index=True
    )[["sector", "near_ath_pct", "total_tickers", "count_near_ath", "date"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    print(result.sort_values("near_ath_pct", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
