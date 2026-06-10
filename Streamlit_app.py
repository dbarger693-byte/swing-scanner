"""Swing trading dashboard. Run with:  streamlit run app.py"""

import pandas as pd
import streamlit as st
import engine as C
from engine import run_scan

st.set_page_config(page_title="Swing Scanner", layout="wide")

st.title("Swing Trading Platform — MVP")
st.caption(
    "Daily-bar system · yfinance data (survivorship-biased — live results will be "
    "worse than backtest) · probabilities are historical frequencies, not forecasts. "
    "Not investment advice."
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_scan(refresh: bool):
    return run_scan(refresh=refresh, check_earnings=True)


col1, col2 = st.columns([1, 4])
with col1:
    refresh = st.button("Refresh data & rescan", type="primary")
with col2:
    st.write("")

with st.spinner("Scanning universe — first run downloads ~3 years of data and may take a few minutes..."):
    if refresh:
        cached_scan.clear()
    res = cached_scan(refresh)

regime = res["regime"]
st.markdown(
    f"**Market regime:** {'🟢 Bull (SPY > 200 SMA)' if regime == 'bull' else '🔴 Bear (SPY < 200 SMA)'} "
    f" · **Data as of:** {res['as_of']:%Y-%m-%d} · **Universe:** {len(C.UNIVERSE)} tickers"
)

tab_sig, tab_watch, tab_perf, tab_journal = st.tabs(
    ["New Signals", "Watchlist", "Performance Analytics", "Trade Journal"]
)

with tab_sig:
    st.subheader(f"Signals scoring ≥ {C.SCORE_THRESHOLD}")
    if res["signals"].empty:
        st.info("No qualifying setups on the latest bar. That is normal — most days produce zero "
                "high-quality signals. Forcing trades is how edges die.")
    else:
        st.dataframe(res["signals"], use_container_width=True, hide_index=True)
        st.caption("If price closes below the Stop, the setup is invalidated — exit and move on. "
                   "P(Target)/P(Stop) are frequencies from the backtest of this setup type "
                   "(regime-matched when sample allows).")

with tab_watch:
    st.subheader("Forming / vetoed setups")
    if res["watchlist"].empty:
        st.info("Nothing forming below the threshold today.")
    else:
        st.dataframe(res["watchlist"], use_container_width=True, hide_index=True)

with tab_perf:
    st.subheader("Backtest — all historical signals in the universe")
    s = res["stats"]
    if s.get("trades", 0) == 0:
        st.warning("No historical trades generated.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", s["trades"])
        m2.metric("Win rate", f"{s['win_rate']:.0%}")
        m3.metric("Profit factor", f"{s['profit_factor']:.2f}")
        m4.metric("Expectancy / trade", f"{s['expectancy']:.2%}")
        m5.metric("Avg hold", f"{s['avg_hold_bars']:.0f} bars")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total return", f"{s['total_return']:.1%}")
        m2.metric("CAGR", f"{s['cagr']:.1%}")
        m3.metric("Max drawdown", f"{s['max_drawdown']:.1%}")
        m4.metric("Sharpe (approx)", f"{s['sharpe']:.2f}")
        m5.metric("Sortino (approx)", f"{s['sortino']:.2f}")
        bench_bits = [f"{k.replace('_buyhold','')} buy-and-hold: {v:.1%}"
                      for k, v in s.items() if k.endswith("_buyhold")]
        if bench_bits:
            st.caption("Benchmarks over the same window — " + " · ".join(bench_bits))
        st.caption(f"Period: {s['period']} · costs {C.COST_PER_TRADE:.2%}/round trip · "
                   f"risk {C.RISK_PER_TRADE:.0%}/trade · Sharpe/Sortino annualized from "
                   "per-trade returns (approximation).")

        st.subheader("Equity curve")
        if len(res["equity"]):
            st.line_chart(res["equity"])

        st.subheader("By setup type and regime")
        tr = res["trades"]
        by = tr.groupby(["setup", "regime"]).agg(
            n=("ret", "size"),
            win_rate=("outcome", lambda x: (x == "target").mean()),
            avg_ret=("ret", "mean"),
            med_hold=("hold_bars", "median"),
        ).round(3)
        st.dataframe(by, use_container_width=True)

with tab_journal:
    st.subheader("Every historical signal the system generated")
    tr = res["trades"]
    if tr.empty:
        st.info("No trades.")
    else:
        j = tr.copy()
        j["ret"] = (j["ret"] * 100).round(2)
        j = j.rename(columns={"ret": "ret %"})
        st.dataframe(
            j.sort_values("entry_date", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.download_button("Download journal CSV", j.to_csv(index=False), "trade_journal.csv")
