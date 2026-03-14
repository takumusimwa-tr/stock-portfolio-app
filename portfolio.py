from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from risk_metrics import historical_expected_shortfall, historical_var

TRADING_DAYS = 252



def annualized_portfolio_return(weights: np.ndarray, mean_daily_returns: pd.Series) -> float:
    # Approximate annualized arithmetic return from daily average returns.
    return float(np.dot(weights, mean_daily_returns) * TRADING_DAYS)



def annualized_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    # Portfolio volatility uses covariance structure across stocks.
    variance = np.dot(weights.T, np.dot(cov_matrix.values * TRADING_DAYS, weights))
    return float(np.sqrt(max(variance, 0.0)))



def portfolio_daily_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    ordered_weights = np.array([weights[c] for c in returns.columns], dtype=float)
    return returns.mul(ordered_weights, axis=1).sum(axis=1)



def run_monte_carlo_optimization(
    returns: pd.DataFrame,
    risk_free_rate: float,
    confidence_level: float,
    n_portfolios: int = 5000,
) -> pd.DataFrame:
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    tickers = list(returns.columns)
    simulations = []

    for _ in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights = weights / weights.sum()

        exp_return = annualized_portfolio_return(weights, mean_daily_returns)
        vol = annualized_portfolio_volatility(weights, cov_matrix)
        sharpe = (exp_return - risk_free_rate) / vol if vol > 0 else np.nan

        weight_dict = {ticker: round(float(weight), 4) for ticker, weight in zip(tickers, weights)}
        daily_port = returns.mul(weights, axis=1).sum(axis=1)
        var_95 = historical_var(daily_port, confidence_level)
        es_95 = historical_expected_shortfall(daily_port, confidence_level)

        row = {
            "expected_return": exp_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "var": var_95,
            "expected_shortfall": es_95,
            "weights": weight_dict,
        }
        for ticker, weight in weight_dict.items():
            row[f"weight_{ticker}"] = weight
        simulations.append(row)

    df = pd.DataFrame(simulations).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    return df



def compute_manual_portfolio_statistics(
    returns: pd.DataFrame,
    weights: dict,
    risk_free_rate: float,
    confidence_level: float,
) -> dict:
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    weight_array = np.array([weights[c] for c in returns.columns], dtype=float)
    daily_port = portfolio_daily_returns(returns, weights)
    ann_return = annualized_portfolio_return(weight_array, mean_daily_returns)
    ann_vol = annualized_portfolio_volatility(weight_array, cov_matrix)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    return {
        "weights": weights,
        "expected_annual_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "value_at_risk": round(float(historical_var(daily_port, confidence_level)), 4),
        "expected_shortfall": round(float(historical_expected_shortfall(daily_port, confidence_level)), 4),
    }



def build_efficient_frontier_chart(mc_df: pd.DataFrame, top5: pd.DataFrame):
    fig = px.scatter(
        mc_df,
        x="volatility",
        y="expected_return",
        color="sharpe_ratio",
        title="Efficient frontier style view of simulated portfolios",
        labels={"volatility": "Annualized volatility", "expected_return": "Expected annual return"},
    )
    fig.add_trace(
        go.Scatter(
            x=top5["volatility"],
            y=top5["expected_return"],
            mode="markers+text",
            text=[f"Top {i+1}" for i in range(len(top5))],
            textposition="top center",
            name="Top 5",
            marker=dict(size=10, symbol="diamond"),
        )
    )
    return fig
