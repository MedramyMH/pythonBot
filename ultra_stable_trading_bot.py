import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Ultra Stable Trading AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Data models
# =============================
@dataclass
class UltraStableSignal:
    symbol: str
    signal_type: str  # BUY / SELL / HOLD
    confidence: float  # 0..1
    strength_score: float  # -1..1
    market_price: float
    pocket_option_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    timestamp: datetime
    key_indicators: Dict
    signal_reasons: List[str]
    reliability_score: float

# =============================
# Helpers
# =============================

def pretty_asset(symbol: str) -> str:
    """Map yfinance symbols to human-readable asset format."""
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-", "/")
    return symbol


def compute_entry_zone(price: float, atr: float) -> Tuple[float, float]:
    """Define a soft entry zone around current price using a fraction of ATR."""
    band = max(atr * 0.15, price * 0.0005)  # at least 5 bps
    return (price - band, price + band)

# =============================
# Pocket Option simulator (pricing + tiny indicator perturbation)
# =============================
class UltraStablePocketOptionSim:
    @staticmethod
    def get_pocket_option_price(market_price: float) -> float:
        variation = np.random.uniform(-0.005, 0.005)
        return float(market_price * (1 + variation))

    @staticmethod
    def get_pocket_option_indicators(market_indicators: Dict) -> Dict:
        po_indicators = {}
        for key, value in market_indicators.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                variation = np.random.uniform(-0.01, 0.01)
                po_indicators[key] = float(value * (1 + variation))
            else:
                po_indicators[key] = value
        return po_indicators

# =============================
# Technicals
# =============================
class UltraStableTechnicalAnalyzer:
    @staticmethod
    def calculate_stable_indicators(df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 50:
            return {}
        indicators = {}
        try:
            close_prices = df['Close']
            # MAs
            indicators['sma_10'] = close_prices.rolling(window=10).mean().iloc[-1]
            indicators['sma_20'] = close_prices.rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = close_prices.rolling(window=50).mean().iloc[-1]

            # RSI
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
            rs = avg_gain / avg_loss
            rsi_series = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi_series.iloc[-1])
            indicators['rsi_trend'] = float(rsi_series.iloc[-1] - rsi_series.iloc[-3]) if len(rsi_series) > 3 else 0.0

            # MACD
            ema_12 = close_prices.ewm(span=12, adjust=False).mean()
            ema_26 = close_prices.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            indicators['macd_histogram'] = float(histogram.iloc[-1])
            indicators['macd_trend'] = float(histogram.iloc[-1] - histogram.iloc[-3]) if len(histogram) > 3 else 0.0

            # Bollinger
            bb_period = 20
            sma_bb = close_prices.rolling(window=bb_period).mean()
            std_bb = close_prices.rolling(window=bb_period).std()
            bb_upper = (sma_bb + (std_bb * 2)).iloc[-1]
            bb_lower = (sma_bb - (std_bb * 2)).iloc[-1]
            indicators['bb_upper'] = float(bb_upper)
            indicators['bb_lower'] = float(bb_lower)
            indicators['bb_middle'] = float(sma_bb.iloc[-1])

            current_price = float(close_prices.iloc[-1])
            indicators['current_price'] = current_price
            bb_width = indicators['bb_upper'] - indicators['bb_lower']
            indicators['bb_position'] = float((current_price - indicators['bb_lower']) / bb_width) if bb_width > 0 else 0.5

            # Momentum
            if len(close_prices) >= 6:
                indicators['momentum_5'] = float((current_price - close_prices.iloc[-6]) / close_prices.iloc[-6] * 100)
            else:
                indicators['momentum_5'] = 0.0

            # ATR
            tr1 = (df['High'] - df['Low']).abs()
            tr2 = (df['High'] - close_prices.shift()).abs()
            tr3 = (df['Low'] - close_prices.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = float(true_range.rolling(window=14).mean().iloc[-1])

            # Volume
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                volume_sma = df['Volume'].rolling(window=20).mean()
                indicators['volume_trend'] = float(df['Volume'].iloc[-1] / max(volume_sma.iloc[-1], 1))
            else:
                indicators['volume_trend'] = 1.0
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
            return {}
        return indicators

# =============================
# Trading Engine (decisioning)
# =============================
class UltraStableTradingEngine:
    def __init__(self):
        self.reliability_threshold = 0.75

    def generate_ultra_stable_signal(self, symbol: str, market_data: Dict, pocket_option_data: Dict) -> UltraStableSignal:
        ind = market_data['indicators']
        if not ind:
            return self._create_neutral_signal(symbol, market_data, pocket_option_data)

        # Scoring: combine RSI, MACD momentum, BB extremes & short momentum
        score = 0.0
        reasons = []

        rsi = ind.get('rsi', 50)
        rsi_trend = ind.get('rsi_trend', 0)
        if rsi < 30 and rsi_trend > 0:
            score += 0.6; reasons.append(f"RSI oversold {rsi:.1f} rising")
        elif rsi > 70 and rsi_trend < 0:
            score -= 0.6; reasons.append(f"RSI overbought {rsi:.1f} falling")

        macd_h = ind.get('macd_histogram', 0)
        macd_tr = ind.get('macd_trend', 0)
        if macd_h > 0 and macd_tr > 0:
            score += 0.5; reasons.append("MACD bullish momentum")
        elif macd_h < 0 and macd_tr < 0:
            score -= 0.5; reasons.append("MACD bearish momentum")

        bb_pos = ind.get('bb_position', 0.5)
        if bb_pos <= 0.1:
            score += 0.4; reasons.append("Near lower Bollinger band")
        elif bb_pos >= 0.9:
            score -= 0.4; reasons.append("Near upper Bollinger band")

        mom5 = ind.get('momentum_5', 0.0)
        if mom5 > 3:
            score += 0.3; reasons.append(f"+Momentum {mom5:.1f}%")
        elif mom5 < -3:
            score -= 0.3; reasons.append(f"-Momentum {mom5:.1f}%")

        # Normalize score to [-1,1]
        strength_score = max(-1.0, min(1.0, score))

        if strength_score > 0.25:
            signal_type = "BUY"
        elif strength_score < -0.25:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"

        # Confidence based on absolute score & simple price agreement
        price = market_data['price']
        po_price = pocket_option_data['price']
        price_agreement = max(0.7, 1.0 - min(abs(price - po_price) / max(price, 1e-6), 0.03) * 10)
        confidence = min(0.95, 0.55 + abs(strength_score) * 0.35) * price_agreement

        # Levels from ATR
        atr = ind.get('atr', price * 0.005)
        entry = price
        rr_mult = 2.0
        if signal_type == "BUY":
            target = price + atr * rr_mult
            stop = price - atr * 1.0
        elif signal_type == "SELL":
            target = price - atr * rr_mult
            stop = price + atr * 1.0
        else:
            target = price
            stop = price
        rrr = abs(target - entry) / max(abs(entry - stop), 1e-6) if stop != entry else 1.0

        return UltraStableSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=float(confidence),
            strength_score=float(strength_score),
            market_price=float(price),
            pocket_option_price=float(po_price),
            entry_price=float(entry),
            target_price=float(target),
            stop_loss=float(stop),
            risk_reward_ratio=float(rrr),
            timestamp=datetime.now(),
            key_indicators={
                'rsi': rsi,
                'macd_histogram': macd_h,
                'bb_position': bb_pos,
                'momentum_5': mom5,
                'atr': atr,
            },
            signal_reasons=reasons if reasons else ["Mixed signals"],
            reliability_score=0.8,
        )

    def _create_neutral_signal(self, symbol: str, market_data: Dict, pocket_option_data: Dict) -> UltraStableSignal:
        price = market_data['price']
        return UltraStableSignal(
            symbol=symbol,
            signal_type="HOLD",
            confidence=0.5,
            strength_score=0.0,
            market_price=float(price),
            pocket_option_price=float(pocket_option_data['price']),
            entry_price=float(price),
            target_price=float(price),
            stop_loss=float(price),
            risk_reward_ratio=1.0,
            timestamp=datetime.now(),
            key_indicators={},
            signal_reasons=["Low reliability - insufficient signal strength"],
            reliability_score=0.5,
        )

# =============================
# Data Manager
# =============================
class UltraStableDataManager:
    def __init__(self):
        self.analyzer = UltraStableTechnicalAnalyzer()

    def get_reliable_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        try:
            # Map timeframe to yfinance interval/period
            if timeframe == '1m':
                interval, period = '1m', '1d'  # 1-minute data (last day)
            elif timeframe == '5m':
                interval, period = '5m', '5d'
            elif timeframe == '15m':
                interval, period = '15m', '30d'
            else:  # fallback daily
                interval, period = '1d', '1y'

            hist = yf.Ticker(symbol).history(period=period, interval=interval)
            if hist.empty or len(hist) < 60:
                return None

            current_price = float(hist['Close'].iloc[-1])
            indicators = self.analyzer.calculate_stable_indicators(hist)
            if not indicators:
                return None

            return {
                'symbol': symbol,
                'price': current_price,
                'indicators': indicators,
                'hist': hist,
            }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

# =============================
# Cached singletons
# =============================
@st.cache_resource
def get_ultra_stable_engine():
    return UltraStableTradingEngine()

@st.cache_resource
def get_ultra_stable_data_manager():
    return UltraStableDataManager()

@st.cache_resource
def get_ultra_stable_pocket_sim():
    return UltraStablePocketOptionSim()

# =============================
# Rendering helpers
# =============================


def decision_block(symbol: str, signal: UltraStableSignal, timeframe: str, contract_minutes: int) -> str:
    asset = pretty_asset(symbol)
    atr = signal.key_indicators.get('atr', max(signal.market_price * 0.005, 1e-6))
    low, high = compute_entry_zone(signal.entry_price, atr)
    lines = [
        f"[Signal] {signal.signal_type}",
        f"[Asset] {asset}",
        f"[Timeframe] {timeframe}",
        f"[Contract Period] {contract_minutes} minutes",
        f"[Entry Zone] {low:.5f} – {high:.5f}",
        f"[Target] {signal.target_price:.5f}",
        f"[Stop Loss] {signal.stop_loss:.5f}",
        f"[Confidence] {signal.confidence*100:.0f}%",
        f"[Reasoning] {', '.join(signal.signal_reasons)}",
    ]
    return "\n".join(lines)

# =============================
# App
# =============================

def main():
    st.title("🎯 Ultra Stable Trading AI")
    st.caption("Auto-refreshing, decision-focused signals (BUY/SELL/HOLD)")

    # Auto refresh control
    refresh_sec = st.sidebar.slider("Auto-refresh every (seconds)", 10, 180, 60, step=5)
    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

    engine = get_ultra_stable_engine()
    data_mgr = get_ultra_stable_data_manager()
    pocket = get_ultra_stable_pocket_sim()

    st.sidebar.header("Assets")
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    selected_stocks = st.sidebar.multiselect("Stocks", stock_symbols, default=[])
    selected_fx = st.sidebar.multiselect("Forex", forex_symbols, default=["EURUSD=X"])  # default to EUR/USD like example
    selected_crypto = st.sidebar.multiselect("Crypto", crypto_symbols, default=[])
    symbols = selected_stocks + selected_fx + selected_crypto

    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1d"], index=0)
    contract_minutes = st.sidebar.number_input("Contract period (minutes)", min_value=1, max_value=60, value=5, step=1)

    min_conf = st.sidebar.slider("Min confidence", 0.5, 0.95, 0.7)
    min_rel = st.sidebar.slider("Min reliability", 0.5, 0.95, 0.75)

    if not symbols:
        st.warning("Select at least one asset from the sidebar.")
        return

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Assets", len(symbols))
    with c2: st.metric("Timeframe", timeframe)
    with c3: st.metric("Contract (min)", contract_minutes)
    with c4: st.metric("Last update", datetime.now().strftime('%H:%M:%S'))

    all_signals: List[Tuple[str, UltraStableSignal, Dict]] = []
    for sym in symbols:
        data = data_mgr.get_reliable_market_data(sym, timeframe=timeframe)
        if not data:
            st.info(f"No recent data for {pretty_asset(sym)} ({timeframe}).")
            continue
        po_price = pocket.get_pocket_option_price(data['price'])
        po_ind = pocket.get_pocket_option_indicators(data['indicators'])
        pocket_data = { 'price': po_price, 'indicators': po_ind }
        sig = engine.generate_ultra_stable_signal(sym, data, pocket_data)
        all_signals.append((sym, sig, data))

    if not all_signals:
        st.warning("No signals could be generated.")
        return

    # Filter
    qualified = [t for t in all_signals if t[1].confidence >= min_conf and t[1].reliability_score >= min_rel]
    if not qualified:
        st.warning("No signals met filters — showing all for review.")
        qualified = all_signals

    st.subheader("📣 Trading Decisions")
    for sym, sig, data in qualified:
        block = decision_block(sym, sig, timeframe, contract_minutes)
        with st.expander(f"{pretty_asset(sym)} — {sig.signal_type}  (Conf: {sig.confidence*100:.0f}%)", expanded=True):
            st.code(block)
            # quick facts row
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Price", f"{sig.market_price:.5f}")
            with k2: st.metric("ATR", f"{sig.key_indicators.get('atr', 0):.5f}")
            with k3: st.metric("RRR", f"{sig.risk_reward_ratio:.2f}")

if __name__ == "__main__":
    main()
