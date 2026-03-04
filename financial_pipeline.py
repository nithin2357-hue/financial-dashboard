"""
Financial Data Pipeline
========================
Author: Nithin Kumar Reddy Panthula
Description:
    A production-style ETL pipeline that fetches live stock market data
    via the Yahoo Finance API, validates, transforms, and exports
    analysis-ready datasets with full audit logging.

Pipeline Stages:
    1. Extract  - Fetch live OHLCV data for multiple tickers via API
    2. Validate - Detect nulls, gaps, anomalies, price integrity issues
    3. Transform - Compute returns, moving averages, volatility, RSI
    4. Load     - Export CSVs, summary report, and audit log
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
PERIOD  = "6mo"          # 6 months of daily data
OUTPUT_DIR = "data/processed"
LOG_DIR    = "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STAGE 1: EXTRACT
# ─────────────────────────────────────────────

def extract(tickers: list, period: str) -> dict:
    """
    Fetch OHLCV data for each ticker via Yahoo Finance API.
    Returns a dict of {ticker: DataFrame}.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: EXTRACT")
    logger.info("=" * 60)
    logger.info(f"Tickers     : {tickers}")
    logger.info(f"Period      : {period}")

    raw_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()
            df["Ticker"] = ticker
            raw_data[ticker] = df
            logger.info(f"Fetched {ticker:6s} : {len(df):4d} rows | "
                        f"{df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")

    logger.info(f"Successfully fetched : {len(raw_data)}/{len(tickers)} tickers")
    return raw_data


# ─────────────────────────────────────────────
# STAGE 2: VALIDATE
# ─────────────────────────────────────────────

def validate(raw_data: dict) -> dict:
    """
    Run data quality checks per ticker:
      - Null/missing values
      - Zero or negative prices
      - Missing trading days (gaps > 5 calendar days)
      - Volume anomalies (zero volume days)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: VALIDATE")
    logger.info("=" * 60)

    total_issues = 0

    for ticker, df in raw_data.items():
        issues = 0

        # Null check
        nulls = df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum()
        if nulls.sum() > 0:
            logger.warning(f"{ticker} | Nulls detected: {nulls[nulls > 0].to_dict()}")
            issues += nulls.sum()
        else:
            logger.info(f"{ticker} | Null check        : PASSED")

        # Negative/zero prices
        bad_prices = df[(df["Close"] <= 0) | (df["Open"] <= 0)]
        if not bad_prices.empty:
            logger.warning(f"{ticker} | Invalid prices    : {len(bad_prices)} rows")
            issues += len(bad_prices)
        else:
            logger.info(f"{ticker} | Price check       : PASSED")

        # Zero volume days
        zero_vol = df[df["Volume"] == 0]
        if not zero_vol.empty:
            logger.warning(f"{ticker} | Zero volume days  : {len(zero_vol)} rows")
            issues += len(zero_vol)
        else:
            logger.info(f"{ticker} | Volume check      : PASSED")

        # Date gaps
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        gaps = date_diffs[date_diffs > 5]
        if not gaps.empty:
            logger.warning(f"{ticker} | Date gaps > 5 days: {len(gaps)} gap(s)")
            issues += len(gaps)
        else:
            logger.info(f"{ticker} | Gap check         : PASSED")

        logger.info(f"{ticker} | Issues flagged    : {issues}")
        total_issues += issues

    logger.info(f"Total issues across all tickers: {total_issues}")
    return raw_data


# ─────────────────────────────────────────────
# STAGE 3: TRANSFORM
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - (100 / (1 + rs))).round(2)


def transform(raw_data: dict) -> dict:
    """
    Per-ticker transformations:
      - Daily returns, cumulative returns
      - 7-day and 30-day moving averages
      - 30-day rolling volatility (annualized)
      - 14-day RSI
      - Trading range (High - Low)

    Aggregate outputs:
      - combined_prices   : all tickers close prices side by side
      - performance_summary: return, volatility, RSI per ticker
      - correlation_matrix : 30-day return correlation between tickers
      - top_movers        : biggest single-day price moves
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: TRANSFORM")
    logger.info("=" * 60)

    enriched = {}
    close_prices = pd.DataFrame()

    for ticker, df in raw_data.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Returns
        df["daily_return_%"]      = (df["Close"].pct_change() * 100).round(4)
        df["cumulative_return_%"] = ((1 + df["Close"].pct_change()).cumprod() - 1) * 100
        df["cumulative_return_%"] = df["cumulative_return_%"].round(4)

        # Moving averages
        df["MA_7"]  = df["Close"].rolling(7).mean().round(2)
        df["MA_30"] = df["Close"].rolling(30).mean().round(2)

        # Volatility (annualized)
        df["volatility_30d_%"] = (
            df["daily_return_%"].rolling(30).std() * np.sqrt(252)
        ).round(2)

        # RSI
        df["RSI_14"] = compute_rsi(df["Close"])

        # Trading range
        df["trading_range"] = (df["High"] - df["Low"]).round(2)

        enriched[ticker] = df
        close_prices = pd.concat(
            [close_prices, df["Close"].rename(ticker).to_frame()],
            axis=1
        )
        close_prices = close_prices[~close_prices.index.duplicated(keep="first")]

        logger.info(f"{ticker} | Cum return: {df['cumulative_return_%'].iloc[-1]:+.2f}% | "
                    f"Volatility: {df['volatility_30d_%'].iloc[-1]:.2f}% | "
                    f"RSI: {df['RSI_14'].iloc[-1]:.1f}")

    # Performance summary
    summary_rows = []
    for ticker, df in enriched.items():
        summary_rows.append({
            "ticker"              : ticker,
            "start_price"         : round(df["Close"].iloc[0], 2),
            "end_price"           : round(df["Close"].iloc[-1], 2),
            "cumulative_return_%" : round(df["cumulative_return_%"].iloc[-1], 2),
            "avg_daily_return_%"  : round(df["daily_return_%"].mean(), 4),
            "volatility_30d_%"    : round(df["volatility_30d_%"].iloc[-1], 2),
            "rsi_14"              : round(df["RSI_14"].iloc[-1], 1),
            "avg_volume"          : int(df["Volume"].mean()),
            "trading_days"        : len(df),
        })
    performance_summary = pd.DataFrame(summary_rows).sort_values(
        "cumulative_return_%", ascending=False
    )

    # Correlation matrix
    correlation_matrix = close_prices.pct_change().corr().round(4)

    # Top movers (biggest single-day absolute moves)
    all_daily = pd.concat([
        df[["Ticker", "daily_return_%", "Close", "Volume"]].assign(date=df.index)
        for df in enriched.values()
    ])
    all_daily = all_daily.reset_index(drop=True)
    top_movers = (
        all_daily.iloc[all_daily["daily_return_%"].abs().sort_values(ascending=False).index]
        .head(20)
        .reset_index(drop=True)
    )

    logger.info(f"Performance summary built for {len(summary_rows)} tickers")
    logger.info(f"Top mover: {top_movers.iloc[0]['Ticker']} "
                f"({top_movers.iloc[0]['daily_return_%']:+.2f}% on "
                f"{str(top_movers.iloc[0]['date'])[:10]})")

    return {
        "enriched"            : enriched,
        "combined_prices"     : close_prices,
        "performance_summary" : performance_summary,
        "correlation_matrix"  : correlation_matrix,
        "top_movers"          : top_movers,
    }


# ─────────────────────────────────────────────
# STAGE 4: LOAD
# ─────────────────────────────────────────────

def load(datasets: dict, output_dir: str) -> None:
    """
    Export all datasets to CSV and generate a JSON summary report.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: LOAD")
    logger.info("=" * 60)

    enriched = datasets.pop("enriched")

    # Export per-ticker enriched data
    for ticker, df in enriched.items():
        path = os.path.join(output_dir, f"{ticker}_enriched.csv")
        df.to_csv(path)
        logger.info(f"Exported: {path} ({len(df)} rows)")

    # Export aggregate datasets
    for name, df in datasets.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path)
        logger.info(f"Exported: {path} ({len(df)} rows)")

    # JSON summary report
    summary = datasets["performance_summary"]
    report = {
        "run_timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers_processed": list(enriched.keys()),
        "top_performer"   : summary.iloc[0]["ticker"],
        "top_return_%"    : float(summary.iloc[0]["cumulative_return_%"]),
        "worst_performer" : summary.iloc[-1]["ticker"],
        "worst_return_%"  : float(summary.iloc[-1]["cumulative_return_%"]),
    }
    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Exported: {report_path}")
    logger.info("All datasets exported successfully.")


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline():
    start = datetime.now()
    logger.info("FINANCIAL DATA PIPELINE STARTED")
    logger.info(f"Timestamp: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        raw      = extract(TICKERS, PERIOD)
        validated = validate(raw)
        datasets = transform(validated)
        load(datasets, OUTPUT_DIR)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()
