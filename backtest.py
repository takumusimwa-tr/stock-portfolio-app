from __future__ import annotations

import numpy as np
import pandas as pd

from risk_metrics import maximum_drawdown

TRADING_DAYS = 252



def backtest_portfolio(returns: pd.DataFrame, benchmark_returns: pd.Series, weights: dict) -> pd.DataFrame:
    ordered_weights = np.array([weights[c] for c in returns.columns], dtype=float)
    portfolio_returns = returns.mul(ordered_weights, axis=1).sum(axis=1)

    bt = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    bt.columns = ["portfolio_returns", "benchmark_returns"]

    bt["portfolio_growth"] = (1 + bt["portfolio_returns"]).cumprod()
    bt["benchmark_growth"] = (1 + bt["benchmark_returns"]).cumprod()

    bt["portfolio_peak"] = bt["portfolio_growth"].cummax()
    bt["benchmark_peak"] = bt["benchmark_growth"].cummax()
    bt["portfolio_drawdown"] = bt["portfolio_growth"] / bt["portfolio_peak"] - 1
    bt["benchmark_drawdown"] = bt["benchmark_growth"] / bt["benchmark_peak"] - 1

    rolling_mean = bt["portfolio_returns"].rolling(63).mean() * TRADING_DAYS
    rolling_vol = bt["portfolio_returns"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    bt["rolling_sharpe_63d"] = rolling_mean / rolling_vol

    return bt



def performance_summary(bt: pd.DataFrame) -> dict:
    port_ret = bt["portfolio_returns"]
    bmk_ret = bt["benchmark_returns"]

    portfolio_annual_return = port_ret.mean() * TRADING_DAYS
    benchmark_annual_return = bmk_ret.mean() * TRADING_DAYS
    portfolio_annual_volatility = port_ret.std() * np.sqrt(TRADING_DAYS)
    benchmark_annual_volatility = bmk_ret.std() * np.sqrt(TRADING_DAYS)

    return {
        "portfolio_annual_return": float(portfolio_annual_return),
        "benchmark_annual_return": float(benchmark_annual_return),
        "portfolio_annual_volatility": float(portfolio_annual_volatility),
        "benchmark_annual_volatility": float(benchmark_annual_volatility),
        "portfolio_max_drawdown": float(maximum_drawdown(bt["portfolio_growth"])),
        "benchmark_max_drawdown": float(maximum_drawdown(bt["benchmark_growth"])),
        "annualized_excess_return": float(portfolio_annual_return - benchmark_annual_return),
    }
