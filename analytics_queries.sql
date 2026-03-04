-- ================================================
-- Financial Data Pipeline - Analytics Queries
-- Author: Nithin Kumar Reddy Panthula
-- ================================================
-- Run these against the exported CSVs using DuckDB,
-- SQLite, or any SQL tool that supports CSV imports.
-- ================================================

-- 1. OVERALL PERFORMANCE RANKING
-- Which stock performed best over the period?
SELECT
    ticker,
    start_price,
    end_price,
    cumulative_return_percent,
    volatility_30d_percent,
    rsi_14,
    trading_days
FROM performance_summary
ORDER BY cumulative_return_percent DESC;


-- 2. RISK vs RETURN ANALYSIS
-- High return + low volatility = best risk-adjusted stocks
SELECT
    ticker,
    cumulative_return_percent,
    volatility_30d_percent,
    ROUND(cumulative_return_percent / NULLIF(volatility_30d_percent, 0), 2) AS sharpe_proxy,
    CASE
        WHEN cumulative_return_percent > 0 AND volatility_30d_percent < 20 THEN 'Low Risk / High Return'
        WHEN cumulative_return_percent > 0 AND volatility_30d_percent >= 20 THEN 'High Risk / High Return'
        WHEN cumulative_return_percent <= 0 AND volatility_30d_percent < 20 THEN 'Low Risk / Low Return'
        ELSE 'High Risk / Low Return'
    END AS risk_category
FROM performance_summary
ORDER BY sharpe_proxy DESC;


-- 3. RSI SIGNAL ANALYSIS
-- Identify overbought (RSI > 70) and oversold (RSI < 30) stocks
SELECT
    ticker,
    rsi_14,
    cumulative_return_percent,
    CASE
        WHEN rsi_14 > 70 THEN 'Overbought - Possible Sell Signal'
        WHEN rsi_14 < 30 THEN 'Oversold - Possible Buy Signal'
        ELSE 'Neutral'
    END AS rsi_signal
FROM performance_summary
ORDER BY rsi_14 DESC;


-- 4. TOP 10 SINGLE-DAY MOVERS
-- Biggest daily price swings across all tickers
SELECT
    ticker,
    date,
    daily_return_percent,
    close,
    volume,
    CASE WHEN daily_return_percent > 0 THEN 'Gain' ELSE 'Loss' END AS direction
FROM top_movers
ORDER BY ABS(daily_return_percent) DESC
LIMIT 10;


-- 5. AVERAGE DAILY VOLUME BY TICKER
-- Which stocks have the most liquidity?
SELECT
    ticker,
    avg_volume,
    trading_days,
    ROUND(avg_volume / 1000000.0, 2) AS avg_volume_millions
FROM performance_summary
ORDER BY avg_volume DESC;


-- 6. MOVING AVERAGE CROSSOVER SIGNALS
-- Detect when 7-day MA crosses above/below 30-day MA (momentum signal)
SELECT
    ticker,
    date,
    close,
    ma_7,
    ma_30,
    CASE
        WHEN ma_7 > ma_30 THEN 'Bullish (MA7 > MA30)'
        WHEN ma_7 < ma_30 THEN 'Bearish (MA7 < MA30)'
        ELSE 'Neutral'
    END AS trend_signal
FROM AAPL_enriched   -- repeat for each ticker
WHERE ma_7 IS NOT NULL AND ma_30 IS NOT NULL
ORDER BY date DESC
LIMIT 30;


-- 7. MONTHLY RETURN SUMMARY
-- Break down returns by month for trend analysis
SELECT
    ticker,
    STRFTIME('%Y-%m', date) AS month,
    ROUND(SUM(daily_return_percent), 2) AS monthly_return_percent,
    COUNT(*) AS trading_days,
    ROUND(AVG(volume) / 1000000.0, 2) AS avg_daily_volume_millions
FROM AAPL_enriched   -- repeat for each ticker
GROUP BY month
ORDER BY month;


-- 8. CORRELATION SUMMARY
-- Which stocks move together? (from correlation_matrix.csv)
-- Values close to 1.0 = highly correlated
-- Values close to 0.0 = independent
SELECT * FROM correlation_matrix;
