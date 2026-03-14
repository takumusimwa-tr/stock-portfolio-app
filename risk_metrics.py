from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

TRADING_DAYS = 252



def historical_var(daily_returns: pd.Series, confidence_level: float = 0.95) -> float:
    # Historical VaR answers: what loss threshold was exceeded only (1 - confidence)% of the time?
    percentile = (1 - confidence_level) * 100
    return float(np.percentile(daily_returns, percentile))



def historical_expected_shortfall(daily_returns: pd.Series, confidence_level: float = 0.95) -> float:
    var_cutoff = historical_var(daily_returns, confidence_level)
    tail = daily_returns[daily_returns <= var_cutoff]
    if tail.empty:
        return float(var_cutoff)
    return float(tail.mean())



def maximum_drawdown(growth_series: pd.Series) -> float:
    running_peak = growth_series.cummax()
    drawdown = growth_series / running_peak - 1
    return float(drawdown.min())



def portfolio_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1]
    var_b = np.var(aligned["benchmark"])
    return float(cov / var_b) if var_b > 0 else np.nan



def classify_sharpe(sharpe: float) -> str:
    if sharpe < 0.5:
        return "weak"
    if sharpe < 1.0:
        return "moderate"
    if sharpe < 1.5:
        return "good"
    return "strong"



def build_metric_explanations(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: dict,
    risk_free_rate: float,
    confidence_level: float,
) -> dict:
    ordered_weights = np.array([weights[c] for c in returns.columns], dtype=float)
    portfolio_ret = returns.mul(ordered_weights, axis=1).sum(axis=1)
    ann_return = portfolio_ret.mean() * TRADING_DAYS
    ann_vol = portfolio_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    growth = (1 + portfolio_ret).cumprod()
    var_val = historical_var(portfolio_ret, confidence_level)
    es_val = historical_expected_shortfall(portfolio_ret, confidence_level)
    beta = portfolio_beta(portfolio_ret, benchmark_returns)
    mdd = maximum_drawdown(growth)

    return {
        "Sharpe Ratio": {
            "value": f"{sharpe:.3f}",
            "method": "(Annualized portfolio return - risk-free rate) / annualized volatility",
            "definition": "Sharpe ratio measures how much return the portfolio produced for each unit of total risk taken.",
            "interpretation": f"This looks {classify_sharpe(sharpe)} on a risk-adjusted basis. Higher is generally better, but it still depends on investor goals.",
        },
        "Value at Risk (VaR)": {
            "value": f"{var_val:.2%}",
            "method": f"Historical simulation at the {int(confidence_level*100)}% confidence level using daily returns.",
            "definition": "VaR estimates a loss threshold that was only breached in the worst few historical trading days.",
            "interpretation": f"At roughly the {int(confidence_level*100)}% level, daily losses worse than this were rare in the sample. It is a threshold, not a worst-case guarantee.",
        },
        "Expected Shortfall / Expected Loss": {
            "value": f"{es_val:.2%}",
            "method": "Average of returns that are worse than the historical VaR cutoff.",
            "definition": "Expected Shortfall estimates the average loss on the bad days beyond the VaR threshold.",
            "interpretation": "This helps describe tail risk more fully than VaR because it focuses on what the worst slice of outcomes actually looked like.",
        },
        "Maximum Drawdown": {
            "value": f"{mdd:.2%}",
            "method": "Minimum value of cumulative growth divided by previous peak minus 1.",
            "definition": "Maximum drawdown is the worst peak-to-trough decline in the backtested growth path.",
            "interpretation": "This is useful for answering the emotionally brutal question: how bad did the ride get before recovery?",
        },
        "Volatility": {
            "value": f"{ann_vol:.2%}",
            "method": "Standard deviation of daily returns annualized using sqrt(252).",
            "definition": "Volatility measures how widely returns moved around their average value.",
            "interpretation": "Higher volatility usually means a bumpier ride. It is not the same thing as permanent loss, but it is part of risk.",
        },
        "Beta vs S&P 500": {
            "value": f"{beta:.3f}",
            "method": "Covariance(portfolio, benchmark) / Variance(benchmark)",
            "definition": "Beta measures how sensitive the portfolio has been to broad market moves relative to the S&P 500.",
            "interpretation": "A beta near 1 means market-like sensitivity, below 1 means lower sensitivity, and above 1 means larger swings than the benchmark.",
        },
    }



def correlation_heatmap(returns: pd.DataFrame):
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")
    return fig
