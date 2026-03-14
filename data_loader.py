from __future__ import annotations

from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

BENCHMARK_TICKER = "^GSPC"

# Small built-in universe so the app works out of the box with visible NYSE logic on the frontend.
UNIVERSE_CSV = """ticker,name,exchange,sector,style_bucket
JPM,JPMorgan Chase & Co.,NYSE,Financials,balanced
XOM,Exxon Mobil Corporation,NYSE,Energy,income
KO,The Coca-Cola Company,NYSE,Consumer Staples,defensive
JNJ,Johnson & Johnson,NYSE,Health Care,defensive
PG,Procter & Gamble Co.,NYSE,Consumer Staples,defensive
HD,Home Depot Inc.,NYSE,Consumer Discretionary,balanced
V,Visa Inc.,NYSE,Financials,growth
BAC,Bank of America Corp.,NYSE,Financials,balanced
CVX,Chevron Corporation,NYSE,Energy,income
WMT,Walmart Inc.,NYSE,Consumer Staples,defensive
DIS,Walt Disney Company,NYSE,Communication Services,growth
MCD,McDonald's Corporation,NYSE,Consumer Discretionary,defensive
IBM,International Business Machines,NYSE,Technology,balanced
GE,GE Aerospace,NYSE,Industrials,growth
CAT,Caterpillar Inc.,NYSE,Industrials,balanced
MS,MSCI? no, Morgan Stanley,NYSE,Financials,balanced
GS,Goldman Sachs Group Inc.,NYSE,Financials,balanced
BLK,BlackRock Inc.,NYSE,Financials,balanced
NKE,Nike Inc.,NYSE,Consumer Discretionary,growth
MMM,3M Company,NYSE,Industrials,income
AAPL,Apple Inc.,NASDAQ,Technology,growth
MSFT,Microsoft Corporation,NASDAQ,Technology,growth
NVDA,NVIDIA Corporation,NASDAQ,Technology,growth
AMZN,Amazon.com Inc.,NASDAQ,Consumer Discretionary,growth
"""


def get_default_stock_universe() -> pd.DataFrame:
    df = pd.read_csv(StringIO(UNIVERSE_CSV))
    df["ticker"] = df["ticker"].str.upper().str.strip()
    return df



def lookup_ticker_exchange(ticker: str, universe_df: pd.DataFrame) -> str | None:
    match = universe_df.loc[universe_df["ticker"] == ticker, "exchange"]
    if match.empty:
        return None
    return str(match.iloc[0])



def is_allowed_ticker(ticker: str, exchange: str | None, allow_all_us: bool) -> Tuple[bool, str]:
    if exchange is None:
        if allow_all_us:
            return True, f"{ticker} is not in the built-in universe list, but it is allowed because 'Allow all U.S.-listed stocks' is enabled."
        return False, f"{ticker} is not in the default supported NYSE universe list. Turn on 'Allow all U.S.-listed stocks' if you want to try broader U.S. tickers."

    if exchange == "NYSE":
        return True, f"{ticker} is NYSE-listed and accepted."

    if allow_all_us:
        return True, f"{ticker} is {exchange}-listed, not NYSE-listed, but it is accepted because the broader U.S.-listed setting is enabled."

    return False, f"{ticker} is {exchange}-listed, not NYSE-listed. Example: AAPL is NASDAQ-listed, so it is rejected unless 'Allow all U.S.-listed stocks' is enabled."



def download_price_data(tickers: List[str], start_date, end_date) -> Tuple[pd.DataFrame, pd.Series]:
    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK_TICKER]))
    raw = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("No data returned from the market data source.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            close = raw.xs("Close", axis=1, level=0, drop_level=True)
    else:
        close = raw.copy()

    close = close.dropna(how="all").ffill().dropna(how="any")

    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing:
        raise ValueError(f"Missing data for tickers: {missing}")
    if BENCHMARK_TICKER not in close.columns:
        raise ValueError("Benchmark data for the S&P 500 could not be loaded.")

    prices = close[tickers].copy()
    benchmark = close[BENCHMARK_TICKER].copy()

    if len(prices) < 30:
        raise ValueError("Insufficient historical data. Please use a longer date range.")

    return prices, benchmark
