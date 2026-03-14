import json
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest import backtest_portfolio, performance_summary
from data_loader import (
    BENCHMARK_TICKER,
    download_price_data,
    get_default_stock_universe,
    is_allowed_ticker,
    lookup_ticker_exchange,
)
from ips import build_ips_summary, default_ips_state
from portfolio import (
    build_efficient_frontier_chart,
    compute_manual_portfolio_statistics,
    run_monte_carlo_optimization,
)
from recommender import recommend_stocks_from_ips
from risk_metrics import build_metric_explanations, correlation_heatmap
from utils import (
    format_pct,
    normalize_weights_from_percent,
    safe_json_download_link,
    validate_date_range,
    validate_weight_inputs,
)

st.set_page_config(
    page_title="Portfolio Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_universe():
    return get_default_stock_universe()


def render_landing_page():
    st.title("📈 Portfolio Lab")
    st.caption(
        "A public-facing portfolio simulator for optimization, backtesting, benchmark comparison, and IPS-guided idea generation."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Benchmark", "S&P 500", BENCHMARK_TICKER)
    c2.metric("Optimization", "Monte Carlo", "Top 5 scenarios")
    c3.metric("Risk Metrics", "Sharpe, VaR, ES", "Drawdown + Beta")
    c4.metric("IPS Module", "Built in", "Rule-based stock ideas")

    with st.expander("What this app does", expanded=True):
        st.markdown(
            """
            This tool lets you pick stocks, test custom portfolio weights, or generate optimized portfolios using Monte Carlo simulation.
            It then backtests the portfolio against the **S&P 500** and explains the risk metrics in plain English.

            It also includes an **Investment Policy Statement (IPS)** workflow so a user can define goals, constraints, and preferences,
            receive explainable stock suggestions, and then send those ideas back into the portfolio workflow.
            """
        )

    with st.expander("Market assumptions and important warnings", expanded=True):
        st.markdown(
            """
            - The analysis assumes an **efficient market**.
            - **Transaction costs** are ignored.
            - **Taxes** are ignored.
            - **Custodial and brokerage fees** are ignored.
            - **Slippage** is ignored.
            - Returns are **compounded through time**.
            - Historical performance **does not guarantee future outcomes**.
            - This tool is for **education and portfolio analysis**, not personalized financial advice.
            """
        )



def render_sidebar(universe_df: pd.DataFrame):
    st.sidebar.header("Configuration")
    allow_all_us = st.sidebar.toggle(
        "Allow all U.S.-listed stocks",
        value=False,
        help="Default mode only accepts NYSE-listed stocks. Enable this if you want to allow NASDAQ or other U.S.-listed tickers too.",
    )

    default_start = date(2018, 1, 1)
    default_end = date.today()
    date_range = st.sidebar.date_input(
        "Analysis period",
        value=(default_start, default_end),
        help="Choose the historical period used for optimization inputs and backtesting.",
    )

    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (annual, decimal)",
        min_value=0.0,
        max_value=0.20,
        value=0.03,
        step=0.005,
        help="Used in the Sharpe ratio calculation. Example: 0.03 means 3% annual risk-free rate.",
    )

    num_portfolios = st.sidebar.slider(
        "Monte Carlo portfolios to simulate",
        min_value=1000,
        max_value=25000,
        step=1000,
        value=5000,
    )

    confidence_level = st.sidebar.slider(
        "VaR / ES confidence level",
        min_value=0.90,
        max_value=0.99,
        step=0.01,
        value=0.95,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("NYSE universe guidance")
    st.sidebar.caption(
        "Example: JPM, XOM, and KO are NYSE-listed. AAPL is NASDAQ-listed, so it is rejected unless the toggle above is enabled."
    )

    return allow_all_us, date_range, risk_free_rate, num_portfolios, confidence_level



def render_portfolio_builder(universe_df: pd.DataFrame, allow_all_us: bool):
    st.header("1) Portfolio construction")

    universe_preview = universe_df[["ticker", "name", "exchange", "sector"]].head(20)
    with st.expander("Preview of supported default universe"):
        st.dataframe(universe_preview, use_container_width=True, hide_index=True)

    default_text = st.session_state.get("prefill_tickers", "JPM, XOM, KO")
    tickers_text = st.text_input(
        "Enter ticker symbols separated by commas",
        value=default_text,
        help="Example: JPM, XOM, KO",
    )

    raw_tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    accepted = []
    warnings = []

    for ticker in raw_tickers:
        exchange = lookup_ticker_exchange(ticker, universe_df)
        allowed, reason = is_allowed_ticker(ticker, exchange, allow_all_us)
        if allowed:
            accepted.append(ticker)
        else:
            warnings.append(reason)

    if warnings:
        for w in warnings:
            st.warning(w)

    if not accepted:
        st.info("Enter at least one valid ticker to continue.")
        return []

    st.success(f"Accepted tickers: {', '.join(accepted)}")

    st.subheader("Manual weights")
    st.caption("Enter portfolio weights in percentages. The total must equal 100%.")

    weight_inputs = {}
    cols = st.columns(min(4, len(accepted)))
    for idx, ticker in enumerate(accepted):
        col = cols[idx % len(cols)]
        with col:
            weight_inputs[ticker] = st.number_input(
                f"{ticker} weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(100.0 / len(accepted), 2),
                step=1.0,
                key=f"weight_{ticker}",
            )

    valid_weights, validation_message = validate_weight_inputs(weight_inputs)
    if valid_weights:
        st.success(validation_message)
    else:
        st.error(validation_message)

    st.session_state["accepted_tickers"] = accepted
    st.session_state["manual_weight_inputs"] = weight_inputs
    return accepted



def render_results_tabs(price_data, benchmark_prices, returns, benchmark_returns, risk_free_rate, confidence_level):
    tab1, tab2, tab3, tab4 = st.tabs(["Optimization", "Backtest", "Risk Metrics", "Correlation"])

    with tab1:
        run_mc = st.button("Run Monte Carlo simulation", type="primary", use_container_width=True)
        if run_mc:
            with st.spinner("Running portfolio simulation..."):
                mc_df = run_monte_carlo_optimization(
                    returns=returns,
                    risk_free_rate=risk_free_rate,
                    confidence_level=confidence_level,
                    n_portfolios=st.session_state["num_portfolios"],
                )
                st.session_state["mc_df"] = mc_df

        mc_df = st.session_state.get("mc_df")
        if mc_df is not None and not mc_df.empty:
            top5 = mc_df.nlargest(5, "sharpe_ratio").reset_index(drop=True)
            st.subheader("Top 5 portfolio scenarios")
            display_cols = [c for c in top5.columns if c not in ["weights"]]
            st.dataframe(top5[display_cols], use_container_width=True, hide_index=True)
            frontier_fig = build_efficient_frontier_chart(mc_df, top5)
            st.plotly_chart(frontier_fig, use_container_width=True)

            selected_rank = st.selectbox(
                "Choose one of the top 5 scenarios for backtesting",
                options=list(range(1, len(top5) + 1)),
                format_func=lambda x: f"Scenario {x}",
            )
            selected_row = top5.iloc[selected_rank - 1]
            st.session_state["selected_weights"] = selected_row["weights"]

            csv = top5.drop(columns=["weights"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download top 5 scenarios as CSV",
                data=csv,
                file_name="top_5_portfolios.csv",
                mime="text/csv",
            )
        else:
            st.info("Run the simulation to see the five best portfolios and the efficient frontier.")

        st.markdown("### Manual portfolio check")
        manual_stats = compute_manual_portfolio_statistics(
            returns,
            normalize_weights_from_percent(st.session_state["manual_weight_inputs"]),
            risk_free_rate,
            confidence_level,
        )
        st.json(manual_stats)

    with tab2:
        chosen_weights = st.session_state.get("selected_weights")
        if not chosen_weights:
            chosen_weights = normalize_weights_from_percent(st.session_state["manual_weight_inputs"])
            st.info("No optimized scenario selected yet, so the manual weights are used for the backtest.")

        bt_df = backtest_portfolio(returns, benchmark_returns, chosen_weights)
        summary = performance_summary(bt_df)
        st.session_state["backtest_summary"] = summary
        st.session_state["backtest_df"] = bt_df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio annualized return", format_pct(summary["portfolio_annual_return"]))
        c2.metric("Portfolio volatility", format_pct(summary["portfolio_annual_volatility"]))
        c3.metric("Max drawdown", format_pct(summary["portfolio_max_drawdown"]))
        c4.metric("Benchmark excess return", format_pct(summary["annualized_excess_return"]))

        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["portfolio_growth"], name="Portfolio"))
        perf_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["benchmark_growth"], name="S&P 500"))
        perf_fig.update_layout(title="Compounded growth of $1", yaxis_title="Growth")
        st.plotly_chart(perf_fig, use_container_width=True)

        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["portfolio_drawdown"], name="Portfolio drawdown"))
        dd_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["benchmark_drawdown"], name="Benchmark drawdown"))
        dd_fig.update_layout(title="Drawdowns", yaxis_title="Drawdown")
        st.plotly_chart(dd_fig, use_container_width=True)

        rolling_fig = go.Figure()
        rolling_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["rolling_sharpe_63d"], name="63-day rolling Sharpe"))
        rolling_fig.update_layout(title="Rolling Sharpe ratio", yaxis_title="Sharpe")
        st.plotly_chart(rolling_fig, use_container_width=True)

    with tab3:
        chosen_weights = st.session_state.get("selected_weights") or normalize_weights_from_percent(
            st.session_state["manual_weight_inputs"]
        )
        metric_cards = build_metric_explanations(
            returns=returns,
            benchmark_returns=benchmark_returns,
            weights=chosen_weights,
            risk_free_rate=risk_free_rate,
            confidence_level=confidence_level,
        )
        for title, info in metric_cards.items():
            with st.expander(title, expanded=False):
                st.markdown(f"**Calculated value:** {info['value']}")
                st.markdown(f"**Method / formula:** {info['method']}")
                st.markdown(f"**Plain-English definition:** {info['definition']}")
                st.markdown(f"**Interpretation:** {info['interpretation']}")

    with tab4:
        heatmap_fig = correlation_heatmap(returns)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.caption("Correlation helps show whether your selected stocks tend to move together or differently. Lower correlation can improve diversification.")



def render_ips_module():
    st.header("2) Investment Policy Statement (IPS)")
    st.caption("This section helps translate investor goals and constraints into a structured policy summary.")

    ips_state = st.session_state.get("ips_state", default_ips_state())

    with st.form("ips_form"):
        objective = st.selectbox(
            "Investment objective",
            ["Capital preservation", "Income", "Balanced growth", "Long-term growth"],
            index=["Capital preservation", "Income", "Balanced growth", "Long-term growth"].index(ips_state["investment_objective"]),
        )
        return_objective = st.text_input("Return objective", value=ips_state["return_objective"])
        risk_tolerance = st.selectbox(
            "Risk tolerance", ["Conservative", "Moderate", "Aggressive"], index=["Conservative", "Moderate", "Aggressive"].index(ips_state["risk_tolerance"])
        )
        liquidity_needs = st.text_input("Liquidity needs", value=ips_state["liquidity_needs"])
        time_horizon = st.selectbox(
            "Time horizon", ["Short-term", "Medium-term", "Long-term"], index=["Short-term", "Medium-term", "Long-term"].index(ips_state["time_horizon"])
        )
        tax_considerations = st.text_input("Tax considerations", value=ips_state["tax_considerations"])
        legal_constraints = st.text_input("Legal / regulatory constraints", value=ips_state["legal_constraints"])
        unique_circumstances = st.text_area("Unique circumstances", value=ips_state["unique_circumstances"])
        rebalancing_preferences = st.selectbox(
            "Rebalancing preference", ["Quarterly", "Semi-annual", "Annual", "Threshold-based"],
            index=["Quarterly", "Semi-annual", "Annual", "Threshold-based"].index(ips_state["rebalancing_preferences"]),
        )
        benchmark_preference = st.text_input("Benchmark preference", value=ips_state["benchmark_preference"])
        diversification_limits = st.text_input("Diversification / concentration limits", value=ips_state["diversification_limits"])
        ethical_exclusions = st.text_input("Ethical or sector exclusions", value=ips_state["ethical_exclusions"])

        submitted = st.form_submit_button("Generate IPS summary", type="primary")

    if submitted:
        ips_state = {
            "investment_objective": objective,
            "return_objective": return_objective,
            "risk_tolerance": risk_tolerance,
            "liquidity_needs": liquidity_needs,
            "time_horizon": time_horizon,
            "tax_considerations": tax_considerations,
            "legal_constraints": legal_constraints,
            "unique_circumstances": unique_circumstances,
            "rebalancing_preferences": rebalancing_preferences,
            "benchmark_preference": benchmark_preference,
            "diversification_limits": diversification_limits,
            "ethical_exclusions": ethical_exclusions,
        }
        st.session_state["ips_state"] = ips_state

    if st.session_state.get("ips_state"):
        summary = build_ips_summary(st.session_state["ips_state"])
        st.subheader("Generated IPS summary")
        st.markdown(summary)
        st.download_button(
            "Download IPS summary as text",
            data=summary.encode("utf-8"),
            file_name="investment_policy_statement.txt",
            mime="text/plain",
        )
        st.markdown(safe_json_download_link(st.session_state["ips_state"], "Download IPS as JSON", "ips.json"), unsafe_allow_html=True)



def render_recommendations(universe_df: pd.DataFrame):
    st.header("3) Stock recommendations from the IPS")
    ips_state = st.session_state.get("ips_state")
    if not ips_state:
        st.info("Complete the IPS section first to unlock explainable stock recommendations.")
        return

    recs = recommend_stocks_from_ips(ips_state, universe_df)
    rec_df = pd.DataFrame(recs)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    selected_recs = st.multiselect(
        "Choose recommended stocks to send into the optimizer",
        options=rec_df["ticker"].tolist(),
        default=rec_df["ticker"].head(3).tolist(),
    )

    if st.button("Use selected recommendations in portfolio builder"):
        st.session_state["prefill_tickers"] = ", ".join(selected_recs)
        st.success("Recommendations loaded into the portfolio builder above.")



def main():
    render_landing_page()
    universe_df = cached_universe()

    allow_all_us, date_range, risk_free_rate, num_portfolios, confidence_level = render_sidebar(universe_df)
    st.session_state["num_portfolios"] = num_portfolios

    valid_dates, date_message = validate_date_range(date_range)
    if not valid_dates:
        st.error(date_message)
        st.stop()
    start_date, end_date = date_range

    selected_tickers = render_portfolio_builder(universe_df, allow_all_us)
    render_ips_module()
    render_recommendations(universe_df)

    if not selected_tickers:
        st.stop()

    if st.button("Load historical data", use_container_width=True):
        with st.spinner("Downloading historical data..."):
            try:
                prices, benchmark = download_price_data(selected_tickers, start_date, end_date)
                st.session_state["prices"] = prices
                st.session_state["benchmark_prices"] = benchmark
            except Exception as exc:
                st.error(f"Could not load market data. Please check the tickers or date range. Technical detail: {str(exc)}")

    price_data = st.session_state.get("prices")
    benchmark_prices = st.session_state.get("benchmark_prices")

    if price_data is not None and benchmark_prices is not None:
        st.subheader("Loaded price data")
        st.dataframe(price_data.tail(), use_container_width=True)

        returns = price_data.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        render_results_tabs(price_data, benchmark_prices, returns, benchmark_returns, risk_free_rate, confidence_level)


if __name__ == "__main__":
    main()
