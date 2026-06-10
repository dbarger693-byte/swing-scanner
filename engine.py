'''Swing trading engine — consolidated single file for easy GitHub mobile upload.
See README in the full project zip for design notes and limitations.'''
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ======================================================================
# config.py
# ======================================================================
"""Central configuration. All parameters are conventional, round numbers chosen
a priori — do not tune them to the backtest (see DESIGN_CRITIQUE.md #1)."""

# ---------------- Universe ----------------
# Liquid, large/mid-cap default universe. Swap in your own list, or in the
# production version pull the full Polygon reference universe.
UNIVERSE = [
    # Mega/large tech
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","TSLA","AMD","CRM",
    "ORCL","ADBE","NOW","INTU","QCOM","TXN","MU","AMAT","LRCX","KLAC",
    "PANW","CRWD","SNOW","NET","DDOG","SHOP","UBER","ABNB","PLTR","SMCI",
    # Financials
    "JPM","BAC","WFC","GS","MS","SCHW","BLK","AXP","V","MA","COF","C",
    # Healthcare
    "LLY","UNH","JNJ","ABBV","MRK","PFE","TMO","DHR","ISRG","VRTX","REGN","AMGN",
    # Consumer
    "WMT","COST","HD","LOW","MCD","SBUX","NKE","TGT","TJX","CMG","LULU","DECK",
    # Industrials / Energy
    "CAT","DE","HON","GE","BA","UNP","UPS","LMT","RTX","XOM","CVX","COP","SLB","EOG",
    # Comms / Media / Other
    "NFLX","DIS","CMCSA","T","VZ","TMUS","SPOT","LIN","FCX","NEM","NUE",
]
BENCHMARKS = ["SPY", "QQQ"]

# ---------------- Liquidity filters ----------------
MIN_PRICE = 10.0                 # no penny stocks
MIN_AVG_DOLLAR_VOLUME = 20e6     # 20-day avg dollar volume floor (spread proxy)

# ---------------- Indicators ----------------
EMA_FAST = 20
SMA_MID = 50
SMA_LONG = 200
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
ROC_LEN = 20
REL_STRENGTH_LOOKBACK = 63       # ~3 months vs SPY

# ---------------- Setup parameters ----------------
PULLBACK_RSI_LO, PULLBACK_RSI_HI = 35, 55
PULLBACK_EMA_PROX = 0.02         # low within 2% of 20 EMA
BREAKOUT_LOOKBACK = 20           # bars of prior highs to clear
BREAKOUT_MIN_RELVOL = 1.5
BOUNCE_RSI_MAX = 30
ADX_MIN_TREND = 18

STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 3.0            # 2:1 reward:risk by construction
BOUNCE_STOP_ATR = 1.2
MIN_RR = 1.8                     # discard anything below this after structure adj.
MAX_HOLD_BARS = 20               # ~4 weeks; time-stop exit

# ---------------- Backtest ----------------
COST_PER_TRADE = 0.0015          # 15 bps round trip (slippage + commission)
RISK_PER_TRADE = 0.01            # 1% of equity risked per trade
MAX_CONCURRENT = 10
START_EQUITY = 100_000.0

# ---------------- Probability engine ----------------
MIN_SAMPLE = 30                  # refuse to print probabilities below this

# ---------------- Ranking ----------------
SCORE_THRESHOLD = 60             # only display setups scoring >= this
WATCHLIST_THRESHOLD = 45         # near-misses shown on the watchlist

# ---------------- Event risk ----------------
EARNINGS_VETO_DAYS = 7           # exclude signals with earnings within N days


# ======================================================================
# data.py
# ======================================================================
"""Data layer. MVP uses yfinance (free, daily bars, survivorship-biased — see
DESIGN_CRITIQUE.md #3). Production swaps this module for Polygon.io without
touching the rest of the system: keep the same get_history() contract."""


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.parquet")


def get_history(ticker: str, period: str = "3y", refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV with columns: Open, High, Low, Close, Volume.
    Cached to parquet so repeated scans don't hammer the data source."""
    path = _cache_path(ticker)
    if not refresh and os.path.exists(path):
        df = pd.read_parquet(path)
        if len(df) > 250:
            return df
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(path)
    return df


def get_universe_data(tickers, period: str = "3y", refresh: bool = False) -> dict:
    out = {}
    for t in tickers:
        try:
            df = get_history(t, period=period, refresh=refresh)
            if len(df) > 250:
                out[t] = df
        except Exception as e:  # one bad ticker shouldn't kill the scan
            print(f"[data] skipped {t}: {e}")
    return out


def next_earnings_date(ticker: str):
    """Best-effort earnings date from yfinance. Returns pd.Timestamp or None."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
            if dates:
                return pd.Timestamp(dates[0])
        elif cal is not None and not cal.empty and "Earnings Date" in cal.index:
            return pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass
    return None


# ======================================================================
# indicators.py
# ======================================================================
"""Indicator library. Pure pandas/numpy — no TA-lib dependency.
Every function takes/returns aligned Series; enrich() attaches them all."""



def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def roc(close: pd.Series, n: int = 20) -> pd.Series:
    return close.pct_change(n) * 100


def hist_vol(close: pd.Series, n: int = 20) -> pd.Series:
    """Annualized historical volatility, %."""
    return close.pct_change().rolling(n).std() * np.sqrt(252) * 100


def rel_volume(vol: pd.Series, n: int = 20) -> pd.Series:
    return vol / vol.rolling(n).mean()


def trend_slope(close: pd.Series, n: int = 20) -> pd.Series:
    """Slope of linear regression on log price, annualized %."""
    logp = np.log(close)
    x = np.arange(n)
    x = x - x.mean()
    denom = (x ** 2).sum()

    def _slope(w):
        return (w - w.mean()) @ x / denom

    return logp.rolling(n).apply(_slope, raw=True) * 252 * 100


def accumulation(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Crude institutional-accumulation proxy on daily bars: net up-volume
    share over n days. > 0.55 suggests accumulation. (Approximation — real
    detection needs intraday data; see DESIGN_CRITIQUE.md #5.)"""
    upday = (df["Close"] > df["Close"].shift()).astype(float)
    upvol = (df["Volume"] * upday).rolling(n).sum()
    return (upvol / df["Volume"].rolling(n).sum()).fillna(0.5)


def enrich(df: pd.DataFrame, spy_close: pd.Series | None = None) -> pd.DataFrame:
    """Attach all factor columns to an OHLCV frame."""
    d = df.copy()
    d["ema20"] = ema(d["Close"], EMA_FAST)
    d["sma50"] = sma(d["Close"], SMA_MID)
    d["sma200"] = sma(d["Close"], SMA_LONG)
    d["rsi"] = rsi(d["Close"], RSI_LEN)
    d["macd"], d["macd_sig"], d["macd_hist"] = macd(d["Close"])
    d["adx"] = adx(d, ADX_LEN)
    d["roc"] = roc(d["Close"], ROC_LEN)
    d["atr"] = atr(d, ATR_LEN)
    d["hvol"] = hist_vol(d["Close"])
    d["relvol"] = rel_volume(d["Volume"])
    d["slope"] = trend_slope(d["Close"])
    d["accum"] = accumulation(d)
    d["dollar_vol"] = (d["Close"] * d["Volume"]).rolling(20).mean()
    d["hi20"] = d["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)  # prior highs only
    d["lo20"] = d["Low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    d["range_pct"] = (d["hi20"] - d["lo20"]) / d["Close"]
    # volatility contraction: current ATR% vs ATR% 20 bars ago
    atr_pct = d["atr"] / d["Close"]
    d["vol_contraction"] = atr_pct < atr_pct.shift(20)
    if spy_close is not None:
        spy = spy_close.reindex(d.index).ffill()
        rs = d["Close"].pct_change(REL_STRENGTH_LOOKBACK) - spy.pct_change(REL_STRENGTH_LOOKBACK)
        d["rel_strength"] = rs * 100
    else:
        d["rel_strength"] = 0.0
    return d


def market_regime(spy: pd.DataFrame) -> pd.Series:
    """'bull' when SPY closes above its 200 SMA, else 'bear'."""
    s200 = sma(spy["Close"], SMA_LONG)
    return pd.Series(np.where(spy["Close"] > s200, "bull", "bear"), index=spy.index)


# ======================================================================
# signals.py
# ======================================================================
"""Signal generation. Three setup types, each defined by binary gates
(conditions that must all be true) — no optimized weights. Entries, targets,
and stops are constructed from ATR and structure (prior swing levels)."""



@dataclass
class Signal:
    ticker: str
    date: pd.Timestamp
    setup: str
    entry: float
    target: float
    stop: float
    atr: float
    regime: str
    factors: dict = field(default_factory=dict)

    @property
    def rr(self) -> float:
        risk = self.entry - self.stop
        return (self.target - self.entry) / risk if risk > 0 else 0.0

    @property
    def expected_gain_pct(self) -> float:
        return (self.target / self.entry - 1) * 100

    @property
    def risk_pct(self) -> float:
        return (1 - self.stop / self.entry) * 100


def _liquid(row) -> bool:
    return row["Close"] >= MIN_PRICE and row["dollar_vol"] >= MIN_AVG_DOLLAR_VOLUME


def detect_at(d: pd.DataFrame, i: int, ticker: str, regime: str) -> Signal | None:
    """Evaluate setups on bar i using ONLY data through bar i (no look-ahead).
    d must be an enriched frame (indicators.enrich)."""
    row = d.iloc[i]
    if i < SMA_LONG or not _liquid(row):
        return None
    px, a = float(row["Close"]), float(row["atr"])
    if a <= 0 or pd.isna(a):
        return None

    common = dict(ticker=ticker, date=d.index[i], atr=a, regime=regime)
    factors = {
        "rsi": round(float(row["rsi"]), 1),
        "adx": round(float(row["adx"]), 1),
        "relvol": round(float(row["relvol"]), 2),
        "rel_strength": round(float(row["rel_strength"]), 1),
        "slope": round(float(row["slope"]), 1),
        "accum": round(float(row["accum"]), 2),
        "macd_hist": round(float(row["macd_hist"]), 3),
        "vol_contraction": bool(row["vol_contraction"]),
    }

    uptrend = px > row["sma50"] > row["sma200"]
    ema_rising = row["ema20"] > d["ema20"].iloc[i - 5]

    # ---- 1. Momentum pullback: uptrend, orderly dip into the 20 EMA ----
    if (
        uptrend and ema_rising
        and row["adx"] >= ADX_MIN_TREND
        and PULLBACK_RSI_LO <= row["rsi"] <= PULLBACK_RSI_HI
        and row["Low"] <= row["ema20"] * (1 + PULLBACK_EMA_PROX)
        and row["rel_strength"] > 0
    ):
        stop = min(px - STOP_ATR_MULT * a, float(d["Low"].iloc[i - 5 : i + 1].min()) - 0.1 * a)
        target = max(px + TARGET_ATR_MULT * a, float(row["hi20"]))
        sig = Signal(**common, setup="Momentum pullback", entry=px, target=target, stop=stop, factors=factors)
        return sig if sig.rr >= MIN_RR else None

    # ---- 2. Breakout: clears 20-bar high on volume after contraction ----
    if (
        px > row["hi20"]
        and row["relvol"] >= BREAKOUT_MIN_RELVOL
        and px > row["sma50"]
        and row["sma50"] > row["sma200"]
        and bool(row["vol_contraction"])
    ):
        base_height = float(row["hi20"] - row["lo20"])
        stop = float(row["hi20"]) - 1.0 * a
        target = max(px + 2.5 * a, float(row["hi20"]) + base_height)  # measured move
        sig = Signal(**common, setup="Breakout", entry=px, target=target, stop=stop, factors=factors)
        return sig if sig.rr >= MIN_RR else None

    # ---- 3. Oversold bounce: quality name, washed-out RSI above 200 SMA ----
    if (
        px > row["sma200"]
        and row["rsi"] <= BOUNCE_RSI_MAX
        and row["accum"] >= 0.45
    ):
        stop = px - BOUNCE_STOP_ATR * a
        target = float(row["ema20"]) if row["ema20"] > px else px + 2.5 * a
        sig = Signal(**common, setup="Oversold bounce", entry=px, target=target, stop=stop, factors=factors)
        return sig if sig.rr >= MIN_RR else None

    return None


def scan_history(d: pd.DataFrame, ticker: str, regime: pd.Series) -> list[Signal]:
    """All historical signals for one ticker (used by backtest + probability
    engine). Enforces a cooldown so overlapping signals don't double-count."""
    out, cooldown = [], 0
    reg = regime.reindex(d.index).ffill()
    for i in range(SMA_LONG, len(d)):
        if cooldown > 0:
            cooldown -= 1
            continue
        s = detect_at(d, i, ticker, str(reg.iloc[i]) if not pd.isna(reg.iloc[i]) else "bull")
        if s:
            out.append(s)
            cooldown = 5
    return out


# ======================================================================
# backtest.py
# ======================================================================
"""Backtesting engine.

Honesty rules (DESIGN_CRITIQUE.md #4):
- Signal on close of bar i -> entry at OPEN of bar i+1 (no look-ahead).
- If a bar touches both target and stop, the STOP is assumed to hit first.
- Gap through stop -> filled at the open (worse than the stop), not at the stop.
- Flat round-trip cost deducted from every trade.
- Time stop at MAX_HOLD_BARS closes the trade at that bar's close.
"""



@dataclass
class Trade:
    ticker: str
    setup: str
    regime: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    target: float
    stop: float
    outcome: str            # 'target' | 'stop' | 'time'
    hold_bars: int
    ret: float              # net fractional return after costs


def simulate_signal(d: pd.DataFrame, sig: Signal) -> Trade | None:
    """Walk forward from the signal bar and resolve the trade."""
    idx = d.index.get_loc(sig.date)
    if idx + 1 >= len(d):
        return None  # signal on the last bar — nothing to fill yet
    entry_i = idx + 1
    entry_px = float(d["Open"].iloc[entry_i])
    # re-anchor target/stop to the actual fill (keep original distances)
    tgt = entry_px + (sig.target - sig.entry)
    stp = entry_px - (sig.entry - sig.stop)
    if stp >= entry_px:
        return None

    last = min(entry_i + MAX_HOLD_BARS, len(d) - 1)
    for j in range(entry_i, last + 1):
        bar = d.iloc[j]
        # gap through stop at the open
        if bar["Open"] <= stp:
            exit_px, outcome = float(bar["Open"]), "stop"
        elif bar["Low"] <= stp:                  # stop first (pessimistic)
            exit_px, outcome = stp, "stop"
        elif bar["High"] >= tgt:
            exit_px, outcome = tgt, "target"
        elif j == last:
            exit_px, outcome = float(bar["Close"]), "time"
        else:
            continue
        ret = exit_px / entry_px - 1 - COST_PER_TRADE
        return Trade(
            ticker=sig.ticker, setup=sig.setup, regime=sig.regime,
            signal_date=sig.date, entry_date=d.index[entry_i], exit_date=d.index[j],
            entry=entry_px, exit=exit_px, target=tgt, stop=stp,
            outcome=outcome, hold_bars=j - entry_i + 1, ret=ret,
        )
    return None


def run_backtest(data: dict, signals_by_ticker: dict) -> pd.DataFrame:
    trades = []
    for t, sigs in signals_by_ticker.items():
        d = data[t]
        for s in sigs:
            tr = simulate_signal(d, s)
            if tr:
                trades.append(tr)
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([vars(t) for t in trades]).sort_values("entry_date").reset_index(drop=True)
    return df


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Fixed-fractional sizing: risk RISK_PER_TRADE of current equity per trade,
    capped at MAX_CONCURRENT notional slots. Simplification documented in
    DESIGN_CRITIQUE.md #6."""
    if trades.empty:
        return pd.Series(dtype=float)
    eq = START_EQUITY
    pts = {}
    for _, tr in trades.iterrows():
        risk_frac = (tr["entry"] - tr["stop"]) / tr["entry"]
        if risk_frac <= 0:
            continue
        size = min(eq * RISK_PER_TRADE / risk_frac, eq / MAX_CONCURRENT)
        eq += size * tr["ret"]
        pts[tr["exit_date"]] = eq
    return pd.Series(pts).sort_index()


def metrics(trades: pd.DataFrame, benchmarks: dict | None = None) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    eq = equity_curve(trades)
    years = max((trades["exit_date"].max() - trades["entry_date"].min()).days / 365.25, 1e-9)
    total_ret = eq.iloc[-1] / START_EQUITY - 1 if len(eq) else 0.0
    cagr = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1.0
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() if len(eq) else 0.0
    # Sharpe/Sortino approximated from per-trade returns annualized by average
    # hold (documented approximation — daily MTM curve is the production fix)
    avg_hold = trades["hold_bars"].mean()
    ann = np.sqrt(max(252 / max(avg_hold, 1), 1))
    sharpe = (r.mean() / r.std() * ann) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = (r.mean() / downside * ann) if downside and downside > 0 else 0.0
    out = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_gain": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "expectancy": r.mean(),
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_hold_bars": avg_hold,
        "period": f"{trades['entry_date'].min():%Y-%m-%d} to {trades['exit_date'].max():%Y-%m-%d}",
    }
    if benchmarks:
        for name, px in benchmarks.items():
            px = px.dropna()
            if len(px) > 1:
                out[f"{name}_buyhold"] = px.iloc[-1] / px.iloc[0] - 1
    return out


# ======================================================================
# probability.py
# ======================================================================
"""Probability engine. Probabilities are FREQUENCIES from the historical
backtest of the same setup type (optionally same market regime) — never
guesses. If the sample is below MIN_SAMPLE, no probability is printed.

Hold-time estimates come from the empirical distribution of resolved trades:
expected = median, fast = 10th percentile, slow = 90th percentile.
"""



def build_tables(trades: pd.DataFrame) -> dict:
    """Pre-compute stats per (setup, regime) and per setup overall."""
    tables = {}
    if trades.empty:
        return tables
    groups = [("setup", "regime"), ("setup",)]
    for keys in groups:
        for name, g in trades.groupby(list(keys)):
            key = name if isinstance(name, tuple) else (name,)
            tables[key] = _stats(g)
    return tables


def _stats(g: pd.DataFrame) -> dict:
    n = len(g)
    hits = (g["outcome"] == "target").sum()
    stops = (g["outcome"] == "stop").sum()
    holds = g["hold_bars"]
    win_holds = g.loc[g["outcome"] == "target", "hold_bars"]
    h = win_holds if len(win_holds) >= 5 else holds
    return {
        "n": n,
        "p_target": hits / n,
        "p_stop": stops / n,
        "p_time": 1 - (hits + stops) / n,
        "expected_hold": float(np.median(h)) if len(h) else np.nan,
        "fast_hold": float(np.percentile(h, 10)) if len(h) else np.nan,
        "slow_hold": float(np.percentile(h, 90)) if len(h) else np.nan,
        "avg_ret": g["ret"].mean(),
    }


def lookup(tables: dict, setup: str, regime: str) -> dict:
    """Prefer regime-matched stats; fall back to setup-wide; enforce sample
    minimum. Returns dict with 'valid' flag."""
    for key in [(setup, regime), (setup,)]:
        s = tables.get(key)
        if s and s["n"] >= MIN_SAMPLE:
            return {**s, "valid": True, "scope": "regime" if len(key) == 2 else "all"}
    s = tables.get((setup,))
    if s:
        return {**s, "valid": False, "scope": "insufficient_sample"}
    return {"valid": False, "n": 0, "scope": "no_history"}


# ======================================================================
# ranking.py
# ======================================================================
"""Ranking engine. 0–100 composite score from six subscores. Weights are fixed
a priori (not optimized). Confidence is reported separately from win
probability: confidence measures evidence quality, probability measures
historical frequency."""


WEIGHTS = {
    "trend": 25,
    "momentum": 20,
    "volume": 15,
    "regime": 15,
    "risk_reward": 15,
    "history": 10,
}


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score_signal(sig, prob: dict) -> dict:
    f = sig.factors
    sub = {}

    # Trend quality: ADX strength + positive slope + relative strength
    sub["trend"] = _clip(0.4 * _clip((f["adx"] - 15) / 25)
                         + 0.3 * _clip(f["slope"] / 50)
                         + 0.3 * _clip((f["rel_strength"] + 5) / 15))

    # Momentum quality: setup-appropriate RSI position + MACD histogram sign
    if sig.setup == "Oversold bounce":
        rsi_q = _clip((35 - f["rsi"]) / 15)          # deeper washout = better
    else:
        rsi_q = _clip(1 - abs(f["rsi"] - 50) / 30)    # neutral reset = better
    sub["momentum"] = _clip(0.6 * rsi_q + 0.4 * (1.0 if f["macd_hist"] > 0 else 0.3))

    # Volume confirmation: relative volume + accumulation balance
    sub["volume"] = _clip(0.5 * _clip((f["relvol"] - 0.8) / 1.2)
                          + 0.5 * _clip((f["accum"] - 0.45) / 0.2))

    # Market regime alignment
    sub["regime"] = 1.0 if sig.regime == "bull" else 0.3

    # Risk/reward profile: 2:1 scores 0.5, 4:1 scores 1.0
    sub["risk_reward"] = _clip((sig.rr - 1.0) / 3.0)

    # Historical performance of similar setups
    if prob.get("valid"):
        sub["history"] = _clip((prob["p_target"] - 0.35) / 0.4)
    else:
        sub["history"] = 0.3  # unproven, not zero

    score = sum(WEIGHTS[k] * v for k, v in sub.items())

    # Confidence: evidence quality (sample size, regime match, volume data)
    conf = 0.0
    conf += 40 * _clip(prob.get("n", 0) / 100)
    conf += 30 * (1.0 if prob.get("scope") == "regime" else 0.5 if prob.get("valid") else 0.0)
    conf += 30 * sub["volume"]

    return {"score": round(score, 1), "confidence": round(conf, 0), "subscores": {k: round(v, 2) for k, v in sub.items()}}


# ======================================================================
# scanner.py
# ======================================================================
"""Scanner orchestrator. One call runs the whole pipeline:
data -> indicators -> historical signals -> backtest -> probability tables ->
today's signals -> ranking -> earnings veto -> output tables."""



def run_scan(refresh: bool = False, check_earnings: bool = True) -> dict:
    tickers = UNIVERSE
    raw = get_universe_data(tickers + BENCHMARKS, refresh=refresh)
    if "SPY" not in raw:
        raise RuntimeError("SPY data unavailable — cannot establish market regime.")

    spy = raw["SPY"]
    regime = market_regime(spy)
    current_regime = str(regime.iloc[-1])

    # Enrich + historical signals for every ticker
    enriched, hist_signals = {}, {}
    for t in tickers:
        if t not in raw:
            continue
        d = enrich(raw[t], spy_close=spy["Close"])
        enriched[t] = d
        hist_signals[t] = scan_history(d, t, regime)

    # Backtest all historical signals -> probability tables + performance
    trades = run_backtest(enriched, hist_signals)
    tables = build_tables(trades)
    bench = {b: raw[b]["Close"] for b in BENCHMARKS if b in raw}
    stats = metrics(trades, benchmarks=bench)
    eq = equity_curve(trades)

    # Today's signals = historical signals dated on each ticker's last bar
    rows, watch_rows = [], []
    for t, sigs in hist_signals.items():
        if not sigs:
            continue
        last_bar = enriched[t].index[-1]
        for sig in sigs:
            if sig.date != last_bar:
                continue
            prob = lookup(tables, sig.setup, sig.regime)
            rank = score_signal(sig, prob)
            row = _format_row(sig, prob, rank)
            if check_earnings:
                ed = next_earnings_date(t)
                if ed is not None:
                    days = (ed - pd.Timestamp.now()).days
                    row["Earnings In"] = f"{days}d" if days >= 0 else "—"
                    if 0 <= days <= EARNINGS_VETO_DAYS:
                        row["Flag"] = f"VETO: earnings in {days}d"
                        watch_rows.append(row)
                        continue
            if rank["score"] >= SCORE_THRESHOLD:
                rows.append(row)
            elif rank["score"] >= WATCHLIST_THRESHOLD:
                row["Flag"] = row.get("Flag", "Below display threshold")
                watch_rows.append(row)

    sig_df = pd.DataFrame(rows).sort_values("Score", ascending=False) if rows else pd.DataFrame()
    watch_df = pd.DataFrame(watch_rows).sort_values("Score", ascending=False) if watch_rows else pd.DataFrame()

    return {
        "signals": sig_df,
        "watchlist": watch_df,
        "trades": trades,
        "stats": stats,
        "equity": eq,
        "prob_tables": tables,
        "regime": current_regime,
        "as_of": enriched[next(iter(enriched))].index[-1] if enriched else None,
    }


def _format_row(sig, prob, rank) -> dict:
    row = {
        "Ticker": sig.ticker,
        "Setup": sig.setup,
        "Entry": round(sig.entry, 2),
        "Target": round(sig.target, 2),
        "Stop": round(sig.stop, 2),
        "R/R": round(sig.rr, 2),
        "Score": rank["score"],
        "Confidence": rank["confidence"],
        "Expected Gain %": round(sig.expected_gain_pct, 1),
        "Risk %": round(sig.risk_pct, 1),
        "Regime": sig.regime,
    }
    if prob.get("valid"):
        row["P(Target)"] = f"{prob['p_target']:.0%}"
        row["P(Stop)"] = f"{prob['p_stop']:.0%}"
        row["Sample"] = prob["n"]
        row["Expected Hold"] = f"{prob['expected_hold']:.0f}d"
        row["Fast Scenario"] = f"{prob['fast_hold']:.0f}d"
        row["Slow Scenario"] = f"{prob['slow_hold']:.0f}d"
    else:
        row["P(Target)"] = "n/a (sample < 30)"
        row["P(Stop)"] = "n/a"
        row["Sample"] = prob.get("n", 0)
        row["Expected Hold"] = "n/a"
        row["Fast Scenario"] = "n/a"
        row["Slow Scenario"] = "n/a"
    row["Reason"] = _reason(sig)
    return row


def _reason(sig) -> str:
    f = sig.factors
    if sig.setup == "Momentum pullback":
        return (f"Uptrend (px>50>200 SMA), ADX {f['adx']}, pullback to 20 EMA with RSI {f['rsi']}, "
                f"RS vs SPY +{f['rel_strength']}%.")
    if sig.setup == "Breakout":
        return (f"Cleared 20-bar high on {f['relvol']}x volume after volatility contraction; "
                f"accumulation {f['accum']}.")
    return f"RSI {f['rsi']} washout above 200 SMA, accumulation {f['accum']} supportive of a bounce to the 20 EMA."
