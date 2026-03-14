from __future__ import annotations


def default_ips_state() -> dict:
    return {
        "investment_objective": "Balanced growth",
        "return_objective": "Target a return above inflation and broad equities over time.",
        "risk_tolerance": "Moderate",
        "liquidity_needs": "Low to moderate liquidity needs.",
        "time_horizon": "Long-term",
        "tax_considerations": "Assume taxes are user-specific and reviewed separately.",
        "legal_constraints": "No special legal constraints stated.",
        "unique_circumstances": "No unusual circumstances provided.",
        "rebalancing_preferences": "Annual",
        "benchmark_preference": "S&P 500",
        "diversification_limits": "Avoid heavy concentration in a single issuer or sector.",
        "ethical_exclusions": "None",
    }



def build_ips_summary(ips_state: dict) -> str:
    return f"""
### Investment Policy Statement Summary

**Investment objective:** {ips_state['investment_objective']}

**Return objective:** {ips_state['return_objective']}

**Risk tolerance:** {ips_state['risk_tolerance']}

**Liquidity needs:** {ips_state['liquidity_needs']}

**Time horizon:** {ips_state['time_horizon']}

**Tax considerations:** {ips_state['tax_considerations']}

**Legal / regulatory constraints:** {ips_state['legal_constraints']}

**Unique circumstances:** {ips_state['unique_circumstances']}

**Rebalancing preferences:** {ips_state['rebalancing_preferences']}

**Benchmark preference:** {ips_state['benchmark_preference']}

**Diversification / concentration limits:** {ips_state['diversification_limits']}

**Ethical or sector exclusions:** {ips_state['ethical_exclusions']}

**Interpretation:**
This IPS is a structured summary of the investor's goals, constraints, and preferences. It is used in the app as an input to the stock recommendation engine and as a guide for evaluating whether a portfolio is aligned with the intended investment style.
""".strip()
