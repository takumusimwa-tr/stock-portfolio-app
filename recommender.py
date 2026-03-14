from __future__ import annotations

import pandas as pd



def recommend_stocks_from_ips(ips_state: dict, universe_df: pd.DataFrame) -> list[dict]:
    risk = ips_state["risk_tolerance"]
    objective = ips_state["investment_objective"]
    exclusions = ips_state.get("ethical_exclusions", "").lower()

    df = universe_df.copy()
    df = df[df["exchange"] == "NYSE"]

    if exclusions:
        for sector in ["energy", "financials", "consumer staples", "health care", "technology", "industrials"]:
            if sector in exclusions:
                df = df[df["sector"].str.lower() != sector]

    if risk == "Conservative" or objective in {"Capital preservation", "Income"}:
        candidates = df[df["style_bucket"].isin(["defensive", "income"])].head(6)
        rationale = "Fits a more conservative or income-oriented IPS because the stock is treated as relatively stable, mature, or dividend-friendly."
    elif risk == "Aggressive" or objective == "Long-term growth":
        candidates = df[df["style_bucket"].isin(["growth", "balanced"])].head(6)
        rationale = "Fits a growth-oriented IPS because the stock is treated as more return-seeking, cyclical, or long-horizon friendly."
    else:
        candidates = df[df["style_bucket"].isin(["balanced", "defensive", "growth"])].head(6)
        rationale = "Fits a moderate IPS because it supports diversification across sectors and styles without leaning too far into either defense or aggression."

    results = []
    for _, row in candidates.iterrows():
        results.append(
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "sector": row["sector"],
                "why_it_fits": rationale,
            }
        )
    return results
