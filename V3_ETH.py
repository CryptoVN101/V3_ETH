"""
V3_Live_Alts.py
------------------------------------------------------------
- Logic Source: V3_TAlts.py (ETHUSDT Backtest)
- Structure: V3_BTC.py (Live Trading Template)
- Features:
    1. Auto Trading on Bybit (Hedge Mode, Kelly Sizing dependent on Market State)
    2. Telegram Notifications (Signal + Exit)
    3. Pending Logic for TREND state only (Phase 5)
- Data: M15 + H1 (Closed candle analysis)
"""

import os
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timezone, timedelta
from telegram import Bot
from telegram.constants import ParseMode

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHANNEL_ID     = os.getenv("CHANNEL_ID", "")

# Bybit
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_SYMBOL     = "ETHUSDT"
BINANCE_SYMBOL   = "ETH/USDT"

# ==============================================================================
# FEATURE TOGGLES
# ==============================================================================

ENABLE_BYBIT_TRADING = False
ENABLE_TELEGRAM      = True

LEVERAGE             = 10
MAX_BYBIT_POSITIONS  = 2

# --- Score Thresholds (synced from V3_TAlts) ---
SCORE_STRONG      = 25   # STRONG_TREND: cần 1 điều kiện
SCORE_WEAK        = 45   # WEAK_TREND  : cần ≥2 điều kiện
SCORE_TREND_LONG  = 45   # TREND LONG  : cần ≥2 điều kiện
SCORE_TREND_SHORT = 65   # TREND SHORT : cần ≥3 điều kiện
SCORE_REV         = 70   # Ngưỡng đảo chiều

ENABLE_PHASE_5_PENDING = True  # Chỉ áp dụng cho TREND state

# --- Risk Management (synced from V3_TAlts) ---
SL_ATR_MULTIPLIER = 1.4
MIN_SL            = 0.015
TP_ATR_MULTIPLIER = 2.6
MIN_RR_RATIO      = 1.7

# --- Kelly Sizing (synced from V3_TAlts) ---
USE_KELLY = True
KELLY_FRACTIONS = {
    "STRONG_TREND": 0.1,
    "TREND":        0.02,
    "WEAK_TREND":   0.06,
}
RISK_PER_TRADE = 0.02

# Operational
VN_TZ       = timezone(timedelta(hours=7))
FETCH_LIMIT = 1200

# API Client (Binance data only)
binance_client = ccxt.binance({'options': {'defaultType': 'future'}})

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ETH_V3_Live")

def get_vn_time():
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# INDICATORS (synced from V3_TAlts)
# ==============================================================================

def calc_tsi(prices):
    if len(prices) < 16:
        return 0
    y = np.array(prices)
    x = np.arange(len(y))
    correlation = np.corrcoef(x, y)[0, 1]
    if not np.isnan(correlation):
        return correlation
    return 0

def calculate_vp(df_slice, num_bins=30):
    if len(df_slice) < 10:
        return None, None, None
    p_min, p_max = df_slice['low'].min(), df_slice['high'].max()
    if p_min == p_max:
        return p_min, p_min, p_min
    bins = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    v_profile = np.zeros(num_bins)
    indices = np.clip(np.digitize(df_slice['close'], bins) - 1, 0, num_bins - 1)
    for i, vol in zip(indices, df_slice['volume']):
        v_profile[i] += vol
    poc_idx = np.argmax(v_profile)
    target_vol = v_profile.sum() * 0.7
    current_vol = v_profile[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx
    while current_vol < target_vol:
        prev_v = v_profile[low_idx-1] if low_idx > 0 else 0
        next_v = v_profile[high_idx+1] if high_idx < num_bins-1 else 0
        if prev_v == 0 and next_v == 0:
            break
        if prev_v >= next_v:
            current_vol += prev_v
            low_idx -= 1
        else:
            current_vol += next_v
            high_idx += 1
    return bin_centers[poc_idx], bin_centers[high_idx], bin_centers[low_idx]

def indicators_h1(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k"]         = stoch.iloc[:, 0]
    df["stoch_d"]         = stoch.iloc[:, 1]
    df["stoch_slope"]     = df["stoch_k"].diff(1)
    df["stoch_neg_count"] = (df["stoch_slope"].shift(1) < 0).rolling(4).sum()
    df["stoch_pos_count"] = (df["stoch_slope"].shift(1) > 0).rolling(4).sum()
    df["adx"]             = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["atr_h1"]          = ta.atr(df["high"], df["low"], df["close"], length=24)
    df["atr_h1_ma"]       = df["atr_h1"].rolling(20).mean()
    df["ema200"]          = ta.ema(df["close"], 200)
    df["ema50"]           = ta.ema(df["close"], 50)
    df["ema50_prev"]      = df["ema50"].shift(1)
    df["high_h1"]         = df["high"]; df["low_h1"]          = df["low"]
    df["high_h1_lag6"]    = df["high"].shift(6)
    df["low_h1_lag6"]     = df["low"].shift(6)
    df["tsi_h1"]          = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["kama_h1"]         = ta.kama(df["close"], 10, 2, 20)
    return df

def indicators_m15(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k_m15"]     = stoch.iloc[:, 0]
    df["stoch_d_m15"]     = stoch.iloc[:, 1]
    df["stoch_slope_m15"] = df["stoch_k_m15"].diff()
    df["kama"]            = ta.kama(df["close"], 10, 2, 20)
    df["kama_slope"]      = df["kama"].diff()
    df["mfi"]             = ta.mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["vwap"]            = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["atr"]             = ta.atr(df["high"], df["low"], df["close"], 16)
    df["atr_avg"]         = df["atr"].rolling(20).mean()
    df["tsi"]             = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["volatility_pct"]  = (df["high"] - df["low"]) / df["close"] * 100
    df["max_volatility_2"]= df["volatility_pct"].shift(1).rolling(2).max()
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bbw"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
    poc_l, vah_l, val_l = [], [], []
    for i in range(len(df)):
        if i < 200: poc_l.append(np.nan); vah_l.append(np.nan); val_l.append(np.nan)
        else:
            p, vah, val = calculate_vp(df.iloc[i-200:i])
            poc_l.append(p); vah_l.append(vah); val_l.append(val)
    df["poc"] = poc_l; df["vah"] = vah_l; df["val"] = val_l
    return df

H1_COLS = [
    "stoch_k", "stoch_d", "stoch_slope", "stoch_neg_count", "stoch_pos_count",
    "adx", "atr_h1", "atr_h1_ma", "ema200", "ema50", "ema50_prev",
    "high_h1", "low_h1", "high_h1_lag6", "low_h1_lag6", "tsi_h1", "kama_h1"
]

def process_data(df_m15):
    df = indicators_m15(df_m15)
    df_h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_h1 = indicators_h1(df_h1)
    df_h1_shifted = df_h1[H1_COLS].shift(1)
    df_merged = df.join(df_h1_shifted, how="left").ffill().dropna()
    return df_merged

# ==============================================================================
# SIGNAL ANALYSIS (synced from V3_TAlts)
# ==============================================================================

def analyze_signal(row):
    stoch_slope_h1 = row.get("stoch_slope", 0)
    neg_count_h1, pos_count_h1 = row.get("stoch_neg_count", 0), row.get("stoch_pos_count", 0)
    adx_h1, atr_h1, atr_h1_ma  = row.get("adx", 0), row.get("atr_h1", 0), row.get("atr_h1_ma", 0)
    ema50_h1, ema200_h1         = row.get("ema50", 0), row.get("ema200", 0)
    ema50_prev                  = row.get("ema50_prev", 0)
    hi, lo, hi6, lo6            = row.get("high_h1", 0), row.get("low_h1", 0), row.get("high_h1_lag6", 0), row.get("low_h1_lag6", 0)
    tsi_h1                      = row.get("tsi_h1", 0) if not pd.isna(row.get("tsi_h1")) else 0
    close, kama, kama_slope     = row.get("close", 0), row.get("kama"), row.get("kama_slope", 0)
    k_m15, ks_m15               = row.get("stoch_k_m15", 50), row.get("stoch_slope_m15", 0)
    mfi, vwap, tsi_m15          = row.get("mfi", 50), row.get("vwap", 0), row.get("tsi", 0)
    atr_m15, atr_avg_m15        = row.get("atr", 0), row.get("atr_avg", 0)
    max_vol_2, poc               = row.get("max_volatility_2", 0), row.get("poc")
    bbw                          = row.get("bbw", 0) if not pd.isna(row.get("bbw")) else 0

    if pd.isna(row.get("stoch_k")) or pd.isna(atr_h1):
        return None, 0, 0, "SIDEWAY", None

    # GĐ1: Market State
    ts = 0
    if adx_h1 >= 50: ts += 2
    elif 35 <= adx_h1 < 50: ts += 1
    hh, hl, lh, ll = hi > hi6, lo > lo6, hi < hi6, lo < lo6
    if (hh and hl) or (lh and ll): ts += 2
    elif hh or ll: ts += 1
    if atr_h1 > atr_h1_ma > 0: ts += 1
    ema_dist  = abs(ema50_h1 - ema200_h1) / close if close > 0 else 0
    ema_slope = ema50_h1 - ema50_prev
    if ema_dist > 0.003: ts += 1
    if close > 0 and abs(ema_slope) / close > 0.0003: ts += 1

    if ts == 7:   state = "STRONG_TREND"
    elif ts >= 4: state = "TREND"
    elif ts >= 3: state = "WEAK_TREND"
    else:         state = "SIDEWAY"

    multiplier = {"STRONG_TREND": 1.5, "TREND": 1.5, "WEAK_TREND": 1.0}.get(state, 0)

    # GĐ2: Entry Trigger
    direction = None
    if (stoch_slope_h1 > 0 and neg_count_h1 >= 3) and (kama and close > kama and kama_slope >= 0):
        direction = "LONG"
    elif (stoch_slope_h1 < 0 and pos_count_h1 >= 3) and (kama and close < kama and kama_slope <= 0):
        direction = "SHORT"
    if not direction: return None, 0, 0, state, None

    # GĐ3: Scoring phân nhánh theo state
    score = 0
    if state == "STRONG_TREND":
        min_sc = SCORE_STRONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
        else:
            if poc and close < poc: score += 30
            if atr_m15 > atr_avg_m15 > 0: score += 20

    elif state == "WEAK_TREND":
        min_sc = SCORE_WEAK
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 20
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 20

    else:  # TREND
        min_sc = SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 15
            if k_m15 < 50 and ks_m15 > 0: score += 15
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 15
            if k_m15 > 50 and ks_m15 < 0: score += 15
            if vwap > 0 and close < vwap and (vwap - close)/vwap < 0.003: score += 15

    if score < min_sc: return None, score, multiplier, state, None

    # GĐ4: Extreme Filters
    if (direction == "SHORT" and tsi_m15 < -0.6) or (direction == "LONG" and tsi_m15 > 0.6):
        return None, score, multiplier, state, None
    if max_vol_2 > 2: return None, score, multiplier, state, None
    if (direction == "LONG"  and row.get("stoch_k") > 75 and tsi_h1 > 0.9) or \
       (direction == "SHORT" and row.get("stoch_k") < 25 and tsi_h1 < -0.9):
        return None, score, multiplier, state, None
    if bbw < 0.0035:
        return f"BLOCK_{direction}_BBW_NEN", score, 0, state, None
    if not pd.isna(mfi) and state in ["TREND", "WEAK_TREND"]:
        if direction == "LONG"  and mfi > 70: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
        if direction == "SHORT" and mfi < 30: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
    if state == "SIDEWAY": return f"BLOCK_{direction}", score, 0, state, None

    # GĐ5: Pending — chỉ áp dụng cho TREND
    if ENABLE_PHASE_5_PENDING and state == "TREND":
        time_hm = (row.name.hour, row.name.minute)
        if time_hm in [(5, 0), (7, 15), (13, 45), (20, 0), (21, 0), (21, 30), (22, 0)]:
            target_time = row.name + timedelta(hours=2)
            return f"PENDING_{direction}_2H", score, multiplier, state, target_time

    return direction, score, multiplier, state, None

# ==============================================================================
# TELEGRAM HELPERS
# ==============================================================================

async def send_telegram(bot, text):
    if not ENABLE_TELEGRAM or not bot or not CHANNEL_ID: return
    try: await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)
    except Exception as e: logger.error(f"❌ Tele Error: {e}")

def _min_sc(direction, state):
    """Lấy ngưỡng score tương ứng theo state và direction."""
    if state == "STRONG_TREND": return SCORE_STRONG
    if state == "WEAK_TREND":   return SCORE_WEAK
    return SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG

def format_signal(direction, entry, sl, tp, score, state, extras=""):
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = tp_pct / sl_pct if sl_pct else 0
    emoji = "🟢" if direction == "LONG" else "🔴"
    return f"""
[V3] 🤖 <b>{direction} ETHUSDT</b> {extras}

💰 Entry: <code>{entry:,.2f}</code>
🛑 SL: <code>{sl:,.2f}</code> ({sl_pct:.2f}%)
🎯 TP: <code>{tp:,.2f}</code> ({tp_pct:.2f}%)

📉 Trend State: <b>{state}</b>
📈 R:R = 1:{rr:.1f}
⭐ Score: {score}

⏰ {get_vn_time()}
"""

def format_close(side, result, price, entry, pnl_pct=None):
    if pnl_pct is None:
        pnl_pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    emoji = "🟢" if result == "TP" else ("🛑" if result == "SL" else "🔄")
    return f"""
[V3] {emoji} <b>{result} ETHUSDT</b>

📌 Vị thế: {side}
🚪 Exit: <code>{price:,.2f}</code>
💰 PnL: <code>{pnl_pct:+.2f}%</code>

⏰ {get_vn_time()}
"""

def calc_sl_tp(price, atr_h1):
    sl_dist = max(atr_h1 * SL_ATR_MULTIPLIER, price * MIN_SL)
    tp_dist = max(sl_dist * MIN_RR_RATIO, atr_h1 * TP_ATR_MULTIPLIER)
    return sl_dist, tp_dist

# ==============================================================================
# BOT STATE
# ==============================================================================

class TradeState:
    def __init__(self):
        self.tele_pos     = None
        self.tele_entry   = None
        self.tele_sl      = None
        self.tele_tp      = None
        self.tele_pending = None
        self.bybit_positions = []
        self.bybit_pending   = None

state_tracker = TradeState()

# ==============================================================================
# MAIN SCAN LOOP
# ==============================================================================

async def run_scan(bot, exchange):
    try:
        ohlcv = await asyncio.to_thread(binance_client.fetch_ohlcv, BINANCE_SYMBOL, '15m', limit=FETCH_LIMIT)
        if not ohlcv: return

        df_m15 = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='ms', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        df_m15.set_index('time', inplace=True)

        df = process_data(df_m15)
        if df is None or df.empty or len(df) < 2:
            logger.warning(f"⚠️ Chưa đủ dữ liệu (Len: {len(df) if df is not None else 0})")
            return

        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = last['close']    # Giá đầu nến mới để khớp lệnh
        current_time  = prev.name        # Timestamp nến đã đóng

        direction, score, multiplier, market_state, pending_time = analyze_signal(prev)

    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}")
        return

    # ==========================================================================
    # TELEGRAM LOGIC
    # ==========================================================================

    # --- Kiểm tra thoát lệnh Tele ---
    if state_tracker.tele_pos:
        result = None; exit_price = 0
        if state_tracker.tele_pos == "LONG":
            if prev['high'] >= state_tracker.tele_tp:   result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] <= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        else:
            if prev['low'] <= state_tracker.tele_tp:    result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] >= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        if result:
            await send_telegram(bot, format_close(state_tracker.tele_pos, result, exit_price, state_tracker.tele_entry))
            state_tracker.tele_pos = None

    # --- Mở lệnh mới Tele ---
    if not state_tracker.tele_pos:
        final_dir = direction; final_score = score; final_state = market_state

        if state_tracker.tele_pending:
            p_end = state_tracker.tele_pending['target_time']
            p_dir = state_tracker.tele_pending['dir']
            if current_time >= p_end:
                min_sc = _min_sc(p_dir, final_state)
                final_dir = p_dir if (score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir) else None
                state_tracker.tele_pending = None
            else:
                final_dir = None

        elif final_dir and "PENDING_" in final_dir:
            state_tracker.tele_pending = {'dir': final_dir.split("_")[1], 'target_time': pending_time}
            final_dir = None
        elif final_dir and "BLOCK" in final_dir:
            final_dir = None

        if final_dir:
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if final_dir == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if final_dir == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(final_dir, current_price, sl, tp, final_score, final_state))
            state_tracker.tele_pos   = final_dir
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # --- Reverse Tele ---
    elif state_tracker.tele_pos:
        vwap = prev.get('vwap', 0) if not pd.isna(prev.get('vwap')) else 0
        vwap_ok = True
        if vwap > 0:
            if state_tracker.tele_pos == "LONG"  and direction == "SHORT" and current_price >= vwap: vwap_ok = False
            if state_tracker.tele_pos == "SHORT" and direction == "LONG"  and current_price <= vwap: vwap_ok = False
        if direction and "BLOCK" not in direction and "PENDING" not in direction \
                and direction != state_tracker.tele_pos and score >= SCORE_REV and vwap_ok:
            pnl_pct = (current_price - state_tracker.tele_entry) / state_tracker.tele_entry * 100 \
                      if state_tracker.tele_pos == "LONG" \
                      else (state_tracker.tele_entry - current_price) / state_tracker.tele_entry * 100
            await send_telegram(bot, format_close(state_tracker.tele_pos, "REV", current_price, state_tracker.tele_entry, pnl_pct))
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if direction == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if direction == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(direction, current_price, sl, tp, score, market_state, "(REV)"))
            state_tracker.tele_pos   = direction
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # ==========================================================================
    # BYBIT LOGIC
    # ==========================================================================

    if ENABLE_BYBIT_TRADING and exchange:
        try:
            actual_pos = await asyncio.to_thread(exchange.fetch_positions, [BYBIT_SYMBOL])
            state_tracker.bybit_positions = []
            for p in actual_pos:
                size = float(p.get('contracts', 0) or p.get('size', 0) or 0)
                if size >= 0.01:
                    side = p.get('side', '').upper()
                    d = "LONG" if side == "BUY" else "SHORT"
                    state_tracker.bybit_positions.append({
                        "dir": d, "entry": float(p.get('entryPrice', 0) or 0),
                        "qty": size, "sl": float(p.get('stopLoss', 0) or 0),
                        "tp": float(p.get('takeProfit', 0) or 0)
                    })

            b_dir = None; b_score = score; b_state = market_state

            # Pending filter
            if state_tracker.bybit_pending:
                p_end = state_tracker.bybit_pending['target_time']
                p_dir = state_tracker.bybit_pending['dir']
                if current_time >= p_end:
                    min_sc = _min_sc(p_dir, b_state)
                    if b_score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir:
                        b_dir = p_dir
                    state_tracker.bybit_pending = None
            elif direction and "PENDING_" in direction:
                state_tracker.bybit_pending = {'dir': direction.split("_")[1], 'target_time': pending_time}
            elif direction and "BLOCK" not in direction:
                b_dir = direction

            # Entry
            if b_dir:
                min_sc = _min_sc(b_dir, b_state)
                if b_score >= min_sc:
                    atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
                    sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
                    sl = current_price - sl_dist if b_dir == "LONG" else current_price + sl_dist
                    tp = current_price + tp_dist if b_dir == "LONG" else current_price - tp_dist

                    bal    = await asyncio.to_thread(exchange.fetch_balance)
                    equity = float(bal['USDT']['total'])

                    if USE_KELLY:
                        risk_amount = equity * KELLY_FRACTIONS.get(b_state, 0.02)
                    else:
                        risk_amount = equity * RISK_PER_TRADE * multiplier

                    sl_pct     = sl_dist / current_price
                    notional   = risk_amount / sl_pct if sl_pct > 0 else 0
                    target_qty = notional / current_price if current_price > 0 else 0
                    max_qty    = (equity * 0.95 / current_price) * LEVERAGE
                    target_qty = min(target_qty, max_qty)

                    existing_same     = [p for p in state_tracker.bybit_positions if p["dir"] == b_dir]
                    existing_opposite = [p for p in state_tracker.bybit_positions if p["dir"] != b_dir]

                    side_str = 'buy' if b_dir == 'LONG' else 'sell'
                    pos_idx  = 1 if b_dir == 'LONG' else 2
                    params   = {
                        'positionIdx': pos_idx,
                        'stopLoss':   {'triggerPrice': str(round(sl, 2)), 'type': 'market'},
                        'takeProfit': {'triggerPrice': str(round(tp, 2)), 'type': 'market'}
                    }

                    if existing_opposite:
                        if b_score < SCORE_REV:
                            logger.info(f"⚠️ REV: score={b_score} < {SCORE_REV} → Bỏ qua")
                        else:
                            # BƯỚC 1: Đóng tất cả lệnh ngược chiều
                            for old_pos in existing_opposite:
                                close_side = 'sell' if old_pos['dir'] == 'LONG' else 'buy'
                                close_idx  = 1 if old_pos['dir'] == 'LONG' else 2
                                try:
                                    await asyncio.to_thread(
                                        exchange.create_market_order, BYBIT_SYMBOL,
                                        close_side, old_pos['qty'],
                                        params={'positionIdx': close_idx, 'reduceOnly': True}
                                    )
                                    logger.info(f"🔄 REV: Đóng {old_pos['dir']} @ {current_price:.2f} | Qty: {old_pos['qty']:.4f}")
                                except Exception as e:
                                    logger.error(f"❌ REV Close Error: {e}")

                            # BƯỚC 2: Mở lệnh mới ngược chiều
                            await asyncio.to_thread(
                                exchange.create_market_order, BYBIT_SYMBOL,
                                side_str, target_qty, params=params
                            )
                            logger.info(f"✅ REV: Mở {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

                    elif existing_same:
                        logger.info(f"ℹ️ Đã có lệnh {b_dir} cùng chiều → Bỏ qua")
                    else:
                        if len(state_tracker.bybit_positions) < MAX_BYBIT_POSITIONS:
                            await asyncio.to_thread(exchange.create_market_order, BYBIT_SYMBOL, side_str, target_qty, params=params)
                            logger.info(f"✅ Bybit Mở mới {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

        except Exception as e:
            logger.error(f"❌ Bybit Error: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    bot = Bot(token=TELEGRAM_TOKEN) if (ENABLE_TELEGRAM and TELEGRAM_TOKEN) else None
    exchange = None
    if ENABLE_BYBIT_TRADING and BYBIT_API_KEY:
        exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'options': {'defaultType': 'linear'}
        })

    features = []
    if ENABLE_BYBIT_TRADING: features.append("📊 Bybit Auto-Trade (ETHUSDT)")
    if ENABLE_TELEGRAM:      features.append("📱 Telegram")
    features_str = "\n".join([f"   • {f}" for f in features]) if features else "   • Analysis Only"

    startup_msg = f"""
🚀 <b>BOT ETHUSDT KHỞI ĐỘNG (V3 Alts)</b>

⏱ Timeframe: M15 + H1
🔧 Features:
{features_str}

📌 Config:
   Kelly: STRONG={KELLY_FRACTIONS['STRONG_TREND']*100:.0f}% | TREND={KELLY_FRACTIONS['TREND']*100:.0f}% | WEAK={KELLY_FRACTIONS['WEAK_TREND']*100:.0f}%

⏰ {get_vn_time()}
"""
    await send_telegram(bot, startup_msg)

    if exchange:
        for _ in range(3):
            try:
                await asyncio.to_thread(exchange.set_position_mode, True, BYBIT_SYMBOL)
                await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                break
            except Exception as e:
                if "110025" in str(e):
                    try: await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                    except: pass
                    break
                await asyncio.sleep(1)
        try:
            bal = await asyncio.to_thread(exchange.fetch_balance)
            equity = float(bal['USDT']['total'])
            logger.info(f"💰 Startup Balance: {equity:.2f} USDT")
        except: pass

    while True:
        try:
            now  = datetime.now()
            wait = (15 - (now.minute % 15)) * 60 - now.second + 3
            if wait < 10: wait = 1
            logger.info(f"⏳ Waiting {wait}s for next M15 candle...")
            await asyncio.sleep(wait)
            await run_scan(bot, exchange)
        except Exception as e:
            logger.error(f"❌ Main Loop Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())

# ==============================================================================
# FEATURE TOGGLES
# ==============================================================================

ENABLE_BYBIT_TRADING = False
ENABLE_TELEGRAM      = True

LEVERAGE             = 10
MAX_BYBIT_POSITIONS  = 2

# --- Score Thresholds (synced from V3_TAlts) ---
SCORE_STRONG      = 25   # STRONG_TREND: cần 1 điều kiện
SCORE_WEAK        = 45   # WEAK_TREND  : cần ≥2 điều kiện
SCORE_TREND_LONG  = 45   # TREND LONG  : cần ≥2 điều kiện
SCORE_TREND_SHORT = 65   # TREND SHORT : cần ≥3 điều kiện
SCORE_REV         = 70   # Ngưỡng đảo chiều

ENABLE_PHASE_5_PENDING = True  # Chỉ áp dụng cho TREND state

# --- Risk Management (synced from V3_TAlts) ---
SL_ATR_MULTIPLIER = 1.4
MIN_SL            = 0.015
TP_ATR_MULTIPLIER = 2.6
MIN_RR_RATIO      = 1.7

# --- Kelly Sizing (synced from V3_TAlts) ---
USE_KELLY = True
KELLY_FRACTIONS = {
    "STRONG_TREND": 0.1,
    "TREND":        0.02,
    "WEAK_TREND":   0.06,
}
RISK_PER_TRADE = 0.02

# Operational
VN_TZ       = timezone(timedelta(hours=7))
FETCH_LIMIT = 1200

# API Client (Binance data only)
binance_client = ccxt.binance({'options': {'defaultType': 'future'}})

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ETH_V3_Live")

def get_vn_time():
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# INDICATORS (synced from V3_TAlts)
# ==============================================================================

def calc_tsi(prices):
    if len(prices) < 16: return 0
    y = np.array(prices)
    x = np.arange(len(y))
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation if not np.isnan(correlation) else 0

def calculate_vp(df_slice, num_bins=30):
    if len(df_slice) < 10: return None, None, None
    p_min, p_max = df_slice['low'].min(), df_slice['high'].max()
    if p_min == p_max: return p_min, p_min, p_min
    bins = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    v_profile = np.zeros(num_bins)
    indices = np.clip(np.digitize(df_slice['close'], bins) - 1, 0, num_bins - 1)
    for i, vol in zip(indices, df_slice['volume']): v_profile[i] += vol
    poc_idx = np.argmax(v_profile)
    target_vol, current_vol = v_profile.sum() * 0.7, v_profile[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx
    while current_vol < target_vol:
        prev_v = v_profile[low_idx-1] if low_idx > 0 else 0
        next_v = v_profile[high_idx+1] if high_idx < num_bins-1 else 0
        if prev_v == 0 and next_v == 0: break
        if prev_v >= next_v: current_vol += prev_v; low_idx -= 1
        else: current_vol += next_v; high_idx += 1
    return bin_centers[poc_idx], bin_centers[high_idx], bin_centers[low_idx]

def indicators_h1(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k"]         = stoch.iloc[:, 0]
    df["stoch_d"]         = stoch.iloc[:, 1]
    df["stoch_slope"]     = df["stoch_k"].diff(1)
    df["stoch_neg_count"] = (df["stoch_slope"].shift(1) < 0).rolling(4).sum()
    df["stoch_pos_count"] = (df["stoch_slope"].shift(1) > 0).rolling(4).sum()
    df["adx"]             = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["atr_h1"]          = ta.atr(df["high"], df["low"], df["close"], length=24)
    df["atr_h1_ma"]       = df["atr_h1"].rolling(20).mean()
    df["ema200"]          = ta.ema(df["close"], 200)
    df["ema50"]           = ta.ema(df["close"], 50)
    df["ema50_prev"]      = df["ema50"].shift(1)
    df["high_h1"]         = df["high"]; df["low_h1"]          = df["low"]
    df["high_h1_lag6"]    = df["high"].shift(6)
    df["low_h1_lag6"]     = df["low"].shift(6)
    df["tsi_h1"]          = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["kama_h1"]         = ta.kama(df["close"], 10, 2, 20)
    return df

def indicators_m15(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k_m15"]     = stoch.iloc[:, 0]
    df["stoch_d_m15"]     = stoch.iloc[:, 1]
    df["stoch_slope_m15"] = df["stoch_k_m15"].diff()
    df["kama"]            = ta.kama(df["close"], 10, 2, 20)
    df["kama_slope"]      = df["kama"].diff()
    df["mfi"]             = ta.mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["vwap"]            = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["atr"]             = ta.atr(df["high"], df["low"], df["close"], 16)
    df["atr_avg"]         = df["atr"].rolling(20).mean()
    df["tsi"]             = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["volatility_pct"]  = (df["high"] - df["low"]) / df["close"] * 100
    df["max_volatility_2"]= df["volatility_pct"].shift(1).rolling(2).max()
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bbw"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
    poc_l, vah_l, val_l = [], [], []
    for i in range(len(df)):
        if i < 200: poc_l.append(np.nan); vah_l.append(np.nan); val_l.append(np.nan)
        else:
            p, vah, val = calculate_vp(df.iloc[i-200:i])
            poc_l.append(p); vah_l.append(vah); val_l.append(val)
    df["poc"] = poc_l; df["vah"] = vah_l; df["val"] = val_l
    return df

H1_COLS = [
    "stoch_k", "stoch_d", "stoch_slope", "stoch_neg_count", "stoch_pos_count",
    "adx", "atr_h1", "atr_h1_ma", "ema200", "ema50", "ema50_prev",
    "high_h1", "low_h1", "high_h1_lag6", "low_h1_lag6", "tsi_h1", "kama_h1"
]

def process_data(df_m15):
    df = indicators_m15(df_m15)
    df_h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_h1 = indicators_h1(df_h1)
    df_h1_shifted = df_h1[H1_COLS].shift(1)
    df_merged = df.join(df_h1_shifted, how="left").ffill().dropna()
    return df_merged

# ==============================================================================
# SIGNAL ANALYSIS (synced from V3_TAlts)
# ==============================================================================

def analyze_signal(row):
    stoch_slope_h1 = row.get("stoch_slope", 0)
    neg_count_h1, pos_count_h1 = row.get("stoch_neg_count", 0), row.get("stoch_pos_count", 0)
    adx_h1, atr_h1, atr_h1_ma  = row.get("adx", 0), row.get("atr_h1", 0), row.get("atr_h1_ma", 0)
    ema50_h1, ema200_h1         = row.get("ema50", 0), row.get("ema200", 0)
    ema50_prev                  = row.get("ema50_prev", 0)
    hi, lo, hi6, lo6            = row.get("high_h1", 0), row.get("low_h1", 0), row.get("high_h1_lag6", 0), row.get("low_h1_lag6", 0)
    tsi_h1                      = row.get("tsi_h1", 0) if not pd.isna(row.get("tsi_h1")) else 0
    close, kama, kama_slope     = row.get("close", 0), row.get("kama"), row.get("kama_slope", 0)
    k_m15, ks_m15               = row.get("stoch_k_m15", 50), row.get("stoch_slope_m15", 0)
    mfi, vwap, tsi_m15          = row.get("mfi", 50), row.get("vwap", 0), row.get("tsi", 0)
    atr_m15, atr_avg_m15        = row.get("atr", 0), row.get("atr_avg", 0)
    max_vol_2, poc               = row.get("max_volatility_2", 0), row.get("poc")
    bbw                          = row.get("bbw", 0) if not pd.isna(row.get("bbw")) else 0

    if pd.isna(row.get("stoch_k")) or pd.isna(atr_h1):
        return None, 0, 0, "SIDEWAY", None

    # GĐ1: Market State
    ts = 0
    if adx_h1 >= 50: ts += 2
    elif 35 <= adx_h1 < 50: ts += 1
    hh, hl, lh, ll = hi > hi6, lo > lo6, hi < hi6, lo < lo6
    if (hh and hl) or (lh and ll): ts += 2
    elif hh or ll: ts += 1
    if atr_h1 > atr_h1_ma > 0: ts += 1
    ema_dist  = abs(ema50_h1 - ema200_h1) / close if close > 0 else 0
    ema_slope = ema50_h1 - ema50_prev
    if ema_dist > 0.003: ts += 1
    if close > 0 and abs(ema_slope) / close > 0.0003: ts += 1

    if ts == 7:   state = "STRONG_TREND"
    elif ts >= 4: state = "TREND"
    elif ts >= 3: state = "WEAK_TREND"
    else:         state = "SIDEWAY"

    multiplier = {"STRONG_TREND": 1.5, "TREND": 1.5, "WEAK_TREND": 1.0}.get(state, 0)

    # GĐ2: Entry Trigger
    direction = None
    if (stoch_slope_h1 > 0 and neg_count_h1 >= 3) and (kama and close > kama and kama_slope >= 0):
        direction = "LONG"
    elif (stoch_slope_h1 < 0 and pos_count_h1 >= 3) and (kama and close < kama and kama_slope <= 0):
        direction = "SHORT"
    if not direction: return None, 0, 0, state, None

    # GĐ3: Scoring phân nhánh theo state
    score = 0
    if state == "STRONG_TREND":
        min_sc = SCORE_STRONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
        else:
            if poc and close < poc: score += 30
            if atr_m15 > atr_avg_m15 > 0: score += 20

    elif state == "WEAK_TREND":
        min_sc = SCORE_WEAK
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 20
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 20

    else:  # TREND
        min_sc = SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 15
            if k_m15 < 50 and ks_m15 > 0: score += 15
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 15
            if k_m15 > 50 and ks_m15 < 0: score += 15
            if vwap > 0 and close < vwap and (vwap - close)/vwap < 0.003: score += 15

    if score < min_sc: return None, score, multiplier, state, None

    # GĐ4: Extreme Filters
    if (direction == "SHORT" and tsi_m15 < -0.6) or (direction == "LONG" and tsi_m15 > 0.6):
        return None, score, multiplier, state, None
    if max_vol_2 > 2: return None, score, multiplier, state, None
    if (direction == "LONG"  and row.get("stoch_k") > 75 and tsi_h1 > 0.9) or \
       (direction == "SHORT" and row.get("stoch_k") < 25 and tsi_h1 < -0.9):
        return None, score, multiplier, state, None
    if bbw < 0.0035:
        return f"BLOCK_{direction}_BBW_NEN", score, 0, state, None
    if not pd.isna(mfi) and state in ["TREND", "WEAK_TREND"]:
        if direction == "LONG"  and mfi > 70: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
        if direction == "SHORT" and mfi < 30: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
    if state == "SIDEWAY": return f"BLOCK_{direction}", score, 0, state, None

    # GĐ5: Pending — chỉ áp dụng cho TREND
    if ENABLE_PHASE_5_PENDING and state == "TREND":
        time_hm = (row.name.hour, row.name.minute)
        if time_hm in [(5, 0), (7, 15), (13, 45), (20, 0), (21, 0), (21, 30), (22, 0)]:
            target_time = row.name + timedelta(hours=2)
            return f"PENDING_{direction}_2H", score, multiplier, state, target_time

    return direction, score, multiplier, state, None

# ==============================================================================
# TELEGRAM HELPERS
# ==============================================================================

async def send_telegram(bot, text):
    if not ENABLE_TELEGRAM or not bot or not CHANNEL_ID: return
    try: await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)
    except Exception as e: logger.error(f"❌ Tele Error: {e}")

def _min_sc(direction, state):
    """Lấy ngưỡng score tương ứng theo state và direction."""
    if state == "STRONG_TREND": return SCORE_STRONG
    if state == "WEAK_TREND":   return SCORE_WEAK
    return SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG

def format_signal(direction, entry, sl, tp, score, state, extras=""):
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = tp_pct / sl_pct if sl_pct else 0
    emoji = "🟢" if direction == "LONG" else "🔴"
    return f"""
[V3] 🤖 <b>{direction} ETHUSDT</b> {extras}

💰 Entry: <code>{entry:,.2f}</code>
🛑 SL: <code>{sl:,.2f}</code> ({sl_pct:.2f}%)
🎯 TP: <code>{tp:,.2f}</code> ({tp_pct:.2f}%)

📉 Trend State: <b>{state}</b>
📈 R:R = 1:{rr:.1f}
⭐ Score: {score}

⏰ {get_vn_time()}
"""

def format_close(side, result, price, entry, pnl_pct=None):
    if pnl_pct is None:
        pnl_pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    emoji = "🟢" if result == "TP" else ("🛑" if result == "SL" else "🔄")
    return f"""
[V3] {emoji} <b>{result} ETHUSDT</b>

📌 Vị thế: {side}
🚪 Exit: <code>{price:,.2f}</code>
💰 PnL: <code>{pnl_pct:+.2f}%</code>

⏰ {get_vn_time()}
"""

def calc_sl_tp(price, atr_h1):
    sl_dist = max(atr_h1 * SL_ATR_MULTIPLIER, price * MIN_SL)
    tp_dist = max(sl_dist * MIN_RR_RATIO, atr_h1 * TP_ATR_MULTIPLIER)
    return sl_dist, tp_dist

# ==============================================================================
# BOT STATE
# ==============================================================================

class TradeState:
    def __init__(self):
        self.tele_pos     = None
        self.tele_entry   = None
        self.tele_sl      = None
        self.tele_tp      = None
        self.tele_pending = None
        self.bybit_positions = []
        self.bybit_pending   = None

state_tracker = TradeState()

# ==============================================================================
# MAIN SCAN LOOP
# ==============================================================================

async def run_scan(bot, exchange):
    try:
        ohlcv = await asyncio.to_thread(binance_client.fetch_ohlcv, BINANCE_SYMBOL, '15m', limit=FETCH_LIMIT)
        if not ohlcv: return

        df_m15 = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='ms', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        df_m15.set_index('time', inplace=True)

        df = process_data(df_m15)
        if df is None or df.empty or len(df) < 2:
            logger.warning(f"⚠️ Chưa đủ dữ liệu (Len: {len(df) if df is not None else 0})")
            return

        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = last['close']    # Giá đầu nến mới để khớp lệnh
        current_time  = prev.name        # Timestamp nến đã đóng

        direction, score, multiplier, market_state, pending_time = analyze_signal(prev)

    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}")
        return

    # ==========================================================================
    # TELEGRAM LOGIC
    # ==========================================================================

    # --- Kiểm tra thoát lệnh Tele ---
    if state_tracker.tele_pos:
        result = None; exit_price = 0
        if state_tracker.tele_pos == "LONG":
            if prev['high'] >= state_tracker.tele_tp:   result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] <= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        else:
            if prev['low'] <= state_tracker.tele_tp:    result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] >= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        if result:
            await send_telegram(bot, format_close(state_tracker.tele_pos, result, exit_price, state_tracker.tele_entry))
            state_tracker.tele_pos = None

    # --- Mở lệnh mới Tele ---
    if not state_tracker.tele_pos:
        final_dir = direction; final_score = score; final_state = market_state

        if state_tracker.tele_pending:
            p_end = state_tracker.tele_pending['target_time']
            p_dir = state_tracker.tele_pending['dir']
            if current_time >= p_end:
                min_sc = _min_sc(p_dir, final_state)
                final_dir = p_dir if (score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir) else None
                state_tracker.tele_pending = None
            else:
                final_dir = None

        elif final_dir and "PENDING_" in final_dir:
            state_tracker.tele_pending = {'dir': final_dir.split("_")[1], 'target_time': pending_time}
            final_dir = None
        elif final_dir and "BLOCK" in final_dir:
            final_dir = None

        if final_dir:
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if final_dir == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if final_dir == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(final_dir, current_price, sl, tp, final_score, final_state))
            state_tracker.tele_pos   = final_dir
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # --- Reverse Tele ---
    elif state_tracker.tele_pos:
        vwap = prev.get('vwap', 0) if not pd.isna(prev.get('vwap')) else 0
        vwap_ok = True
        if vwap > 0:
            if state_tracker.tele_pos == "LONG"  and direction == "SHORT" and current_price >= vwap: vwap_ok = False
            if state_tracker.tele_pos == "SHORT" and direction == "LONG"  and current_price <= vwap: vwap_ok = False
        if direction and "BLOCK" not in direction and "PENDING" not in direction \
                and direction != state_tracker.tele_pos and score >= SCORE_REV and vwap_ok:
            pnl_pct = (current_price - state_tracker.tele_entry) / state_tracker.tele_entry * 100 \
                      if state_tracker.tele_pos == "LONG" \
                      else (state_tracker.tele_entry - current_price) / state_tracker.tele_entry * 100
            await send_telegram(bot, format_close(state_tracker.tele_pos, "REV", current_price, state_tracker.tele_entry, pnl_pct))
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if direction == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if direction == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(direction, current_price, sl, tp, score, market_state, "(REV)"))
            state_tracker.tele_pos   = direction
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # ==========================================================================
    # BYBIT LOGIC
    # ==========================================================================

    if ENABLE_BYBIT_TRADING and exchange:
        try:
            actual_pos = await asyncio.to_thread(exchange.fetch_positions, [BYBIT_SYMBOL])
            state_tracker.bybit_positions = []
            for p in actual_pos:
                size = float(p.get('contracts', 0) or p.get('size', 0) or 0)
                if size >= 0.01:
                    side = p.get('side', '').upper()
                    d = "LONG" if side == "BUY" else "SHORT"
                    state_tracker.bybit_positions.append({
                        "dir": d, "entry": float(p.get('entryPrice', 0) or 0),
                        "qty": size, "sl": float(p.get('stopLoss', 0) or 0),
                        "tp": float(p.get('takeProfit', 0) or 0)
                    })

            b_dir = None; b_score = score; b_state = market_state

            # Pending filter
            if state_tracker.bybit_pending:
                p_end = state_tracker.bybit_pending['target_time']
                p_dir = state_tracker.bybit_pending['dir']
                if current_time >= p_end:
                    min_sc = _min_sc(p_dir, b_state)
                    if b_score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir:
                        b_dir = p_dir
                    state_tracker.bybit_pending = None
            elif direction and "PENDING_" in direction:
                state_tracker.bybit_pending = {'dir': direction.split("_")[1], 'target_time': pending_time}
            elif direction and "BLOCK" not in direction:
                b_dir = direction

            # Entry
            if b_dir:
                min_sc = _min_sc(b_dir, b_state)
                if b_score >= min_sc:
                    atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
                    sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
                    sl = current_price - sl_dist if b_dir == "LONG" else current_price + sl_dist
                    tp = current_price + tp_dist if b_dir == "LONG" else current_price - tp_dist

                    bal    = await asyncio.to_thread(exchange.fetch_balance)
                    equity = float(bal['USDT']['total'])

                    if USE_KELLY:
                        risk_amount = equity * KELLY_FRACTIONS.get(b_state, 0.02)
                    else:
                        risk_amount = equity * RISK_PER_TRADE * multiplier

                    sl_pct     = sl_dist / current_price
                    notional   = risk_amount / sl_pct if sl_pct > 0 else 0
                    target_qty = notional / current_price if current_price > 0 else 0
                    max_qty    = (equity * 0.95 / current_price) * LEVERAGE
                    target_qty = min(target_qty, max_qty)

                    existing_same     = [p for p in state_tracker.bybit_positions if p["dir"] == b_dir]
                    existing_opposite = [p for p in state_tracker.bybit_positions if p["dir"] != b_dir]

                    side_str = 'buy' if b_dir == 'LONG' else 'sell'
                    pos_idx  = 1 if b_dir == 'LONG' else 2
                    params   = {
                        'positionIdx': pos_idx,
                        'stopLoss':   {'triggerPrice': str(round(sl, 2)), 'type': 'market'},
                        'takeProfit': {'triggerPrice': str(round(tp, 2)), 'type': 'market'}
                    }

                    if existing_opposite:
                        if b_score < SCORE_REV:
                            logger.info(f"⚠️ REV: score={b_score} < {SCORE_REV} → Bỏ qua")
                        else:
                            # BƯỚC 1: Đóng tất cả lệnh ngược chiều
                            for old_pos in existing_opposite:
                                close_side = 'sell' if old_pos['dir'] == 'LONG' else 'buy'
                                close_idx  = 1 if old_pos['dir'] == 'LONG' else 2
                                try:
                                    await asyncio.to_thread(
                                        exchange.create_market_order, BYBIT_SYMBOL,
                                        close_side, old_pos['qty'],
                                        params={'positionIdx': close_idx, 'reduceOnly': True}
                                    )
                                    logger.info(f"🔄 REV: Đóng {old_pos['dir']} @ {current_price:.2f} | Qty: {old_pos['qty']:.4f}")
                                except Exception as e:
                                    logger.error(f"❌ REV Close Error: {e}")

                            # BƯỚC 2: Mở lệnh mới ngược chiều
                            await asyncio.to_thread(
                                exchange.create_market_order, BYBIT_SYMBOL,
                                side_str, target_qty, params=params
                            )
                            logger.info(f"✅ REV: Mở {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

                    elif existing_same:
                        logger.info(f"ℹ️ Đã có lệnh {b_dir} cùng chiều → Bỏ qua")
                    else:
                        if len(state_tracker.bybit_positions) < MAX_BYBIT_POSITIONS:
                            await asyncio.to_thread(exchange.create_market_order, BYBIT_SYMBOL, side_str, target_qty, params=params)
                            logger.info(f"✅ Bybit Mở mới {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

        except Exception as e:
            logger.error(f"❌ Bybit Error: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    bot = Bot(token=TELEGRAM_TOKEN) if (ENABLE_TELEGRAM and TELEGRAM_TOKEN) else None
    exchange = None
    if ENABLE_BYBIT_TRADING and BYBIT_API_KEY:
        exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'options': {'defaultType': 'linear'}
        })

    features = []
    if ENABLE_BYBIT_TRADING: features.append("📊 Bybit Auto-Trade (ETHUSDT)")
    if ENABLE_TELEGRAM:      features.append("📱 Telegram")
    features_str = "\n".join([f"   • {f}" for f in features]) if features else "   • Analysis Only"

    startup_msg = f"""
🚀 <b>BOT ETHUSDT KHỞI ĐỘNG (V3 Alts)</b>

⏱ Timeframe: M15 + H1
🔧 Features:
{features_str}

📌 Config:
   Kelly: STRONG={KELLY_FRACTIONS['STRONG_TREND']*100:.0f}% | TREND={KELLY_FRACTIONS['TREND']*100:.0f}% | WEAK={KELLY_FRACTIONS['WEAK_TREND']*100:.0f}%

⏰ {get_vn_time()}
"""
    await send_telegram(bot, startup_msg)

    if exchange:
        for _ in range(3):
            try:
                await asyncio.to_thread(exchange.set_position_mode, True, BYBIT_SYMBOL)
                await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                break
            except Exception as e:
                if "110025" in str(e):
                    try: await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                    except: pass
                    break
                await asyncio.sleep(1)
        try:
            bal = await asyncio.to_thread(exchange.fetch_balance)
            equity = float(bal['USDT']['total'])
            logger.info(f"💰 Startup Balance: {equity:.2f} USDT")
        except: pass

    while True:
        try:
            now  = datetime.now()
            wait = (15 - (now.minute % 15)) * 60 - now.second + 3
            if wait < 10: wait = 1
            logger.info(f"⏳ Waiting {wait}s for next M15 candle...")
            await asyncio.sleep(wait)
            await run_scan(bot, exchange)
        except Exception as e:
            logger.error(f"❌ Main Loop Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
    p_min, p_max = df_slice['low'].min(), df_slice['high'].max()
    if p_min == p_max: return p_min, p_min, p_min
    bins = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    v_profile = np.zeros(num_bins)
    indices = np.clip(np.digitize(df_slice['close'], bins) - 1, 0, num_bins - 1)
    for i, vol in zip(indices, df_slice['volume']): v_profile[i] += vol
    poc_idx = np.argmax(v_profile)
    target_vol, current_vol = v_profile.sum() * 0.7, v_profile[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx
    while current_vol < target_vol:
        prev_v = v_profile[low_idx-1] if low_idx > 0 else 0
        next_v = v_profile[high_idx+1] if high_idx < num_bins-1 else 0
        if prev_v == 0 and next_v == 0: break
        if prev_v >= next_v: current_vol += prev_v; low_idx -= 1
        else: current_vol += next_v; high_idx += 1
    return bin_centers[poc_idx], bin_centers[high_idx], bin_centers[low_idx]

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHANNEL_ID     = os.getenv("CHANNEL_ID", "")

# Bybit
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_SYMBOL     = "ETHUSDT"
BINANCE_SYMBOL   = "ETH/USDT"

# ==============================================================================
# FEATURE TOGGLES
# ==============================================================================

ENABLE_BYBIT_TRADING = False
ENABLE_TELEGRAM      = True

LEVERAGE             = 10
MAX_BYBIT_POSITIONS  = 2

# --- Score Thresholds (synced from V3_TAlts) ---
SCORE_STRONG      = 25   # STRONG_TREND: cần 1 điều kiện
SCORE_WEAK        = 45   # WEAK_TREND  : cần ≥2 điều kiện
SCORE_TREND_LONG  = 45   # TREND LONG  : cần ≥2 điều kiện
SCORE_TREND_SHORT = 65   # TREND SHORT : cần ≥3 điều kiện
SCORE_REV         = 70   # Ngưỡng đảo chiều

ENABLE_PHASE_5_PENDING = True  # Chỉ áp dụng cho TREND state

# --- Risk Management (synced from V3_TAlts) ---
SL_ATR_MULTIPLIER = 1.4
MIN_SL            = 0.015
TP_ATR_MULTIPLIER = 2.6
MIN_RR_RATIO      = 1.7

# --- Kelly Sizing (synced from V3_TAlts) ---
USE_KELLY = True
KELLY_FRACTIONS = {
    "STRONG_TREND": 0.1,
    "TREND":        0.02,
    "WEAK_TREND":   0.06,
}
RISK_PER_TRADE = 0.02

# Operational
VN_TZ       = timezone(timedelta(hours=7))
FETCH_LIMIT = 1200

# API Client (Binance data only)
binance_client = ccxt.binance({'options': {'defaultType': 'future'}})

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ETH_V3_Live")

def get_vn_time():
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# INDICATORS (synced from V3_TAlts)
# ==============================================================================

def indicators_h1(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k"]         = stoch.iloc[:, 0]
    df["stoch_d"]         = stoch.iloc[:, 1]
    df["stoch_slope"]     = df["stoch_k"].diff(1)
    df["stoch_neg_count"] = (df["stoch_slope"].shift(1) < 0).rolling(4).sum()
    df["stoch_pos_count"] = (df["stoch_slope"].shift(1) > 0).rolling(4).sum()
    df["adx"]             = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["atr_h1"]          = ta.atr(df["high"], df["low"], df["close"], length=24)
    df["atr_h1_ma"]       = df["atr_h1"].rolling(20).mean()
    df["ema200"]          = ta.ema(df["close"], 200)
    df["ema50"]           = ta.ema(df["close"], 50)
    df["ema50_prev"]      = df["ema50"].shift(1)
    df["high_h1"]         = df["high"]; df["low_h1"]          = df["low"]
    df["high_h1_lag6"]    = df["high"].shift(6)
    df["low_h1_lag6"]     = df["low"].shift(6)
    df["tsi_h1"]          = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["kama_h1"]         = ta.kama(df["close"], 10, 2, 20)
    return df

def indicators_m15(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k_m15"]     = stoch.iloc[:, 0]
    df["stoch_d_m15"]     = stoch.iloc[:, 1]
    df["stoch_slope_m15"] = df["stoch_k_m15"].diff()
    df["kama"]            = ta.kama(df["close"], 10, 2, 20)
    df["kama_slope"]      = df["kama"].diff()
    df["mfi"]             = ta.mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["vwap"]            = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["atr"]             = ta.atr(df["high"], df["low"], df["close"], 16)
    df["atr_avg"]         = df["atr"].rolling(20).mean()
    df["tsi"]             = df["close"].rolling(16).apply(calc_tsi, raw=True)
    df["volatility_pct"]  = (df["high"] - df["low"]) / df["close"] * 100
    df["max_volatility_2"]= df["volatility_pct"].shift(1).rolling(2).max()
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bbw"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
    poc_l, vah_l, val_l = [], [], []
    for i in range(len(df)):
        if i < 200: poc_l.append(np.nan); vah_l.append(np.nan); val_l.append(np.nan)
        else:
            p, vah, val = calculate_vp(df.iloc[i-200:i])
            poc_l.append(p); vah_l.append(vah); val_l.append(val)
    df["poc"] = poc_l; df["vah"] = vah_l; df["val"] = val_l
    return df

H1_COLS = [
    "stoch_k", "stoch_d", "stoch_slope", "stoch_neg_count", "stoch_pos_count",
    "adx", "atr_h1", "atr_h1_ma", "ema200", "ema50", "ema50_prev",
    "high_h1", "low_h1", "high_h1_lag6", "low_h1_lag6", "tsi_h1", "kama_h1"
]

def process_data(df_m15):
    df = indicators_m15(df_m15)
    df_h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_h1 = indicators_h1(df_h1)
    df_h1_shifted = df_h1[H1_COLS].shift(1)
    df_merged = df.join(df_h1_shifted, how="left").ffill().dropna()
    return df_merged

# ==============================================================================
# SIGNAL ANALYSIS (synced from V3_TAlts)
# ==============================================================================

def analyze_signal(row):
    stoch_slope_h1 = row.get("stoch_slope", 0)
    neg_count_h1, pos_count_h1 = row.get("stoch_neg_count", 0), row.get("stoch_pos_count", 0)
    adx_h1, atr_h1, atr_h1_ma  = row.get("adx", 0), row.get("atr_h1", 0), row.get("atr_h1_ma", 0)
    ema50_h1, ema200_h1         = row.get("ema50", 0), row.get("ema200", 0)
    ema50_prev                  = row.get("ema50_prev", 0)
    hi, lo, hi6, lo6            = row.get("high_h1", 0), row.get("low_h1", 0), row.get("high_h1_lag6", 0), row.get("low_h1_lag6", 0)
    tsi_h1                      = row.get("tsi_h1", 0) if not pd.isna(row.get("tsi_h1")) else 0
    close, kama, kama_slope     = row.get("close", 0), row.get("kama"), row.get("kama_slope", 0)
    k_m15, ks_m15               = row.get("stoch_k_m15", 50), row.get("stoch_slope_m15", 0)
    mfi, vwap, tsi_m15          = row.get("mfi", 50), row.get("vwap", 0), row.get("tsi", 0)
    atr_m15, atr_avg_m15        = row.get("atr", 0), row.get("atr_avg", 0)
    max_vol_2, poc               = row.get("max_volatility_2", 0), row.get("poc")
    bbw                          = row.get("bbw", 0) if not pd.isna(row.get("bbw")) else 0

    if pd.isna(row.get("stoch_k")) or pd.isna(atr_h1):
        return None, 0, 0, "SIDEWAY", None

    # GĐ1: Market State
    ts = 0
    if adx_h1 >= 50: ts += 2
    elif 35 <= adx_h1 < 50: ts += 1
    hh, hl, lh, ll = hi > hi6, lo > lo6, hi < hi6, lo < lo6
    if (hh and hl) or (lh and ll): ts += 2
    elif hh or ll: ts += 1
    if atr_h1 > atr_h1_ma > 0: ts += 1
    ema_dist  = abs(ema50_h1 - ema200_h1) / close if close > 0 else 0
    ema_slope = ema50_h1 - ema50_prev
    if ema_dist > 0.003: ts += 1
    if close > 0 and abs(ema_slope) / close > 0.0003: ts += 1

    if ts == 7:   state = "STRONG_TREND"
    elif ts >= 4: state = "TREND"
    elif ts >= 3: state = "WEAK_TREND"
    else:         state = "SIDEWAY"

    multiplier = {"STRONG_TREND": 1.5, "TREND": 1.5, "WEAK_TREND": 1.0}.get(state, 0)

    # GĐ2: Entry Trigger
    direction = None
    if (stoch_slope_h1 > 0 and neg_count_h1 >= 3) and (kama and close > kama and kama_slope >= 0):
        direction = "LONG"
    elif (stoch_slope_h1 < 0 and pos_count_h1 >= 3) and (kama and close < kama and kama_slope <= 0):
        direction = "SHORT"
    if not direction: return None, 0, 0, state, None

    # GĐ3: Scoring phân nhánh theo state
    score = 0
    if state == "STRONG_TREND":
        min_sc = SCORE_STRONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
        else:
            if poc and close < poc: score += 30
            if atr_m15 > atr_avg_m15 > 0: score += 20

    elif state == "WEAK_TREND":
        min_sc = SCORE_WEAK
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 20
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 20

    else:  # TREND
        min_sc = SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG
        if direction == "LONG":
            if atr_m15 > atr_avg_m15 > 0: score += 30
            if vwap > 0 and close > vwap and (close - vwap)/vwap < 0.003: score += 25
            if tsi_m15 > 0: score += 15
            if k_m15 < 50 and ks_m15 > 0: score += 15
        else:
            if poc and close < poc: score += 30
            if tsi_m15 < 0: score += 25
            if atr_m15 > atr_avg_m15 > 0: score += 15
            if k_m15 > 50 and ks_m15 < 0: score += 15
            if vwap > 0 and close < vwap and (vwap - close)/vwap < 0.003: score += 15

    if score < min_sc: return None, score, multiplier, state, None

    # GĐ4: Extreme Filters
    if (direction == "SHORT" and tsi_m15 < -0.6) or (direction == "LONG" and tsi_m15 > 0.6):
        return None, score, multiplier, state, None
    if max_vol_2 > 2: return None, score, multiplier, state, None
    if (direction == "LONG"  and row.get("stoch_k") > 75 and tsi_h1 > 0.9) or \
       (direction == "SHORT" and row.get("stoch_k") < 25 and tsi_h1 < -0.9):
        return None, score, multiplier, state, None
    if bbw < 0.0035:
        return f"BLOCK_{direction}_BBW_NEN", score, 0, state, None
    if not pd.isna(mfi) and state in ["TREND", "WEAK_TREND"]:
        if direction == "LONG"  and mfi > 70: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
        if direction == "SHORT" and mfi < 30: return f"BLOCK_{direction}_FOMO_MFI", score, 0, state, None
    if state == "SIDEWAY": return f"BLOCK_{direction}", score, 0, state, None

    # GĐ5: Pending — chỉ áp dụng cho TREND
    if ENABLE_PHASE_5_PENDING and state == "TREND":
        time_hm = (row.name.hour, row.name.minute)
        if time_hm in [(5, 0), (7, 15), (13, 45), (20, 0), (21, 0), (21, 30), (22, 0)]:
            target_time = row.name + timedelta(hours=2)
            return f"PENDING_{direction}_2H", score, multiplier, state, target_time

    return direction, score, multiplier, state, None

# ==============================================================================
# TELEGRAM HELPERS
# ==============================================================================

async def send_telegram(bot, text):
    if not ENABLE_TELEGRAM or not bot or not CHANNEL_ID: return
    try: await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)
    except Exception as e: logger.error(f"❌ Tele Error: {e}")

def _min_sc(direction, state):
    """Lấy ngưỡng score tương ứng theo state và direction."""
    if state == "STRONG_TREND": return SCORE_STRONG
    if state == "WEAK_TREND":   return SCORE_WEAK
    return SCORE_TREND_SHORT if direction == "SHORT" else SCORE_TREND_LONG

def format_signal(direction, entry, sl, tp, score, state, extras=""):
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = tp_pct / sl_pct if sl_pct else 0
    emoji = "🟢" if direction == "LONG" else "🔴"
    return f"""
[V3] 🤖 <b>{direction} ETHUSDT</b> {extras}

💰 Entry: <code>{entry:,.2f}</code>
🛑 SL: <code>{sl:,.2f}</code> ({sl_pct:.2f}%)
🎯 TP: <code>{tp:,.2f}</code> ({tp_pct:.2f}%)

📉 Trend State: <b>{state}</b>
📈 R:R = 1:{rr:.1f}
⭐ Score: {score}

⏰ {get_vn_time()}
"""

def format_close(side, result, price, entry, pnl_pct=None):
    if pnl_pct is None:
        pnl_pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    emoji = "🟢" if result == "TP" else ("🛑" if result == "SL" else "🔄")
    return f"""
[V3] {emoji} <b>{result} ETHUSDT</b>

📌 Vị thế: {side}
🚪 Exit: <code>{price:,.2f}</code>
💰 PnL: <code>{pnl_pct:+.2f}%</code>

⏰ {get_vn_time()}
"""

def calc_sl_tp(price, atr_h1):
    sl_dist = max(atr_h1 * SL_ATR_MULTIPLIER, price * MIN_SL)
    tp_dist = max(sl_dist * MIN_RR_RATIO, atr_h1 * TP_ATR_MULTIPLIER)
    return sl_dist, tp_dist

# ==============================================================================
# BOT STATE
# ==============================================================================

class TradeState:
    def __init__(self):
        self.tele_pos     = None
        self.tele_entry   = None
        self.tele_sl      = None
        self.tele_tp      = None
        self.tele_pending = None
        self.bybit_positions = []
        self.bybit_pending   = None

state_tracker = TradeState()

# ==============================================================================
# MAIN SCAN LOOP
# ==============================================================================

async def run_scan(bot, exchange):
    try:
        ohlcv = await asyncio.to_thread(binance_client.fetch_ohlcv, BINANCE_SYMBOL, '15m', limit=FETCH_LIMIT)
        if not ohlcv: return

        df_m15 = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='ms', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
        df_m15.set_index('time', inplace=True)

        df = process_data(df_m15)
        if df is None or df.empty or len(df) < 2:
            logger.warning(f"⚠️ Chưa đủ dữ liệu (Len: {len(df) if df is not None else 0})")
            return

        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = last['close']    # Giá đầu nến mới để khớp lệnh
        current_time  = prev.name        # Timestamp nến đã đóng

        direction, score, multiplier, market_state, pending_time = analyze_signal(prev)

    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}")
        return

    # ==========================================================================
    # TELEGRAM LOGIC
    # ==========================================================================

    # --- Kiểm tra thoát lệnh Tele ---
    if state_tracker.tele_pos:
        result = None; exit_price = 0
        if state_tracker.tele_pos == "LONG":
            if prev['high'] >= state_tracker.tele_tp:   result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] <= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        else:
            if prev['low'] <= state_tracker.tele_tp:    result, exit_price = "TP", state_tracker.tele_tp
            elif prev['close'] >= state_tracker.tele_sl: result, exit_price = "SL", state_tracker.tele_sl
        if result:
            await send_telegram(bot, format_close(state_tracker.tele_pos, result, exit_price, state_tracker.tele_entry))
            state_tracker.tele_pos = None

    # --- Mở lệnh mới Tele ---
    if not state_tracker.tele_pos:
        final_dir = direction; final_score = score; final_state = market_state

        if state_tracker.tele_pending:
            p_end = state_tracker.tele_pending['target_time']
            p_dir = state_tracker.tele_pending['dir']
            if current_time >= p_end:
                min_sc = _min_sc(p_dir, final_state)
                final_dir = p_dir if (score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir) else None
                state_tracker.tele_pending = None
            else:
                final_dir = None

        elif final_dir and "PENDING_" in final_dir:
            state_tracker.tele_pending = {'dir': final_dir.split("_")[1], 'target_time': pending_time}
            final_dir = None
        elif final_dir and "BLOCK" in final_dir:
            final_dir = None

        if final_dir:
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if final_dir == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if final_dir == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(final_dir, current_price, sl, tp, final_score, final_state))
            state_tracker.tele_pos   = final_dir
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # --- Reverse Tele ---
    elif state_tracker.tele_pos:
        vwap = prev.get('vwap', 0) if not pd.isna(prev.get('vwap')) else 0
        vwap_ok = True
        if vwap > 0:
            if state_tracker.tele_pos == "LONG"  and direction == "SHORT" and current_price >= vwap: vwap_ok = False
            if state_tracker.tele_pos == "SHORT" and direction == "LONG"  and current_price <= vwap: vwap_ok = False
        if direction and "BLOCK" not in direction and "PENDING" not in direction \
                and direction != state_tracker.tele_pos and score >= SCORE_REV and vwap_ok:
            pnl_pct = (current_price - state_tracker.tele_entry) / state_tracker.tele_entry * 100 \
                      if state_tracker.tele_pos == "LONG" \
                      else (state_tracker.tele_entry - current_price) / state_tracker.tele_entry * 100
            await send_telegram(bot, format_close(state_tracker.tele_pos, "REV", current_price, state_tracker.tele_entry, pnl_pct))
            atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
            sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
            sl = current_price - sl_dist if direction == "LONG" else current_price + sl_dist
            tp = current_price + tp_dist if direction == "LONG" else current_price - tp_dist
            await send_telegram(bot, format_signal(direction, current_price, sl, tp, score, market_state, "(REV)"))
            state_tracker.tele_pos   = direction
            state_tracker.tele_entry = current_price
            state_tracker.tele_sl    = sl
            state_tracker.tele_tp    = tp
            state_tracker.tele_pending = None

    # ==========================================================================
    # BYBIT LOGIC
    # ==========================================================================

    if ENABLE_BYBIT_TRADING and exchange:
        try:
            actual_pos = await asyncio.to_thread(exchange.fetch_positions, [BYBIT_SYMBOL])
            state_tracker.bybit_positions = []
            for p in actual_pos:
                size = float(p.get('contracts', 0) or p.get('size', 0) or 0)
                if size >= 0.01:
                    side = p.get('side', '').upper()
                    d = "LONG" if side == "BUY" else "SHORT"
                    state_tracker.bybit_positions.append({
                        "dir": d, "entry": float(p.get('entryPrice', 0) or 0),
                        "qty": size, "sl": float(p.get('stopLoss', 0) or 0),
                        "tp": float(p.get('takeProfit', 0) or 0)
                    })

            b_dir = None; b_score = score; b_state = market_state

            # Pending filter
            if state_tracker.bybit_pending:
                p_end = state_tracker.bybit_pending['target_time']
                p_dir = state_tracker.bybit_pending['dir']
                if current_time >= p_end:
                    min_sc = _min_sc(p_dir, b_state)
                    if b_score >= min_sc and "BLOCK" not in (direction or "") and direction == p_dir:
                        b_dir = p_dir
                    state_tracker.bybit_pending = None
            elif direction and "PENDING_" in direction:
                state_tracker.bybit_pending = {'dir': direction.split("_")[1], 'target_time': pending_time}
            elif direction and "BLOCK" not in direction:
                b_dir = direction

            # Entry
            if b_dir:
                min_sc = _min_sc(b_dir, b_state)
                if b_score >= min_sc:
                    atr_h1_val = prev.get('atr_h1', 0) if not pd.isna(prev.get('atr_h1')) else 0
                    sl_dist, tp_dist = calc_sl_tp(current_price, atr_h1_val)
                    sl = current_price - sl_dist if b_dir == "LONG" else current_price + sl_dist
                    tp = current_price + tp_dist if b_dir == "LONG" else current_price - tp_dist

                    bal    = await asyncio.to_thread(exchange.fetch_balance)
                    equity = float(bal['USDT']['total'])

                    if USE_KELLY:
                        risk_amount = equity * KELLY_FRACTIONS.get(b_state, 0.02)
                    else:
                        risk_amount = equity * RISK_PER_TRADE * multiplier

                    sl_pct     = sl_dist / current_price
                    notional   = risk_amount / sl_pct if sl_pct > 0 else 0
                    target_qty = notional / current_price if current_price > 0 else 0
                    max_qty    = (equity * 0.95 / current_price) * LEVERAGE
                    target_qty = min(target_qty, max_qty)

                    existing_same     = [p for p in state_tracker.bybit_positions if p["dir"] == b_dir]
                    existing_opposite = [p for p in state_tracker.bybit_positions if p["dir"] != b_dir]

                    side_str = 'buy' if b_dir == 'LONG' else 'sell'
                    pos_idx  = 1 if b_dir == 'LONG' else 2
                    params   = {
                        'positionIdx': pos_idx,
                        'stopLoss':   {'triggerPrice': str(round(sl, 2)), 'type': 'market'},
                        'takeProfit': {'triggerPrice': str(round(tp, 2)), 'type': 'market'}
                    }

                    if existing_opposite:
                        if b_score < SCORE_REV:
                            logger.info(f"⚠️ REV: score={b_score} < {SCORE_REV} → Bỏ qua")
                        else:
                            # BƯỚC 1: Đóng tất cả lệnh ngược chiều
                            for old_pos in existing_opposite:
                                close_side = 'sell' if old_pos['dir'] == 'LONG' else 'buy'
                                close_idx  = 1 if old_pos['dir'] == 'LONG' else 2
                                try:
                                    await asyncio.to_thread(
                                        exchange.create_market_order, BYBIT_SYMBOL,
                                        close_side, old_pos['qty'],
                                        params={'positionIdx': close_idx, 'reduceOnly': True}
                                    )
                                    logger.info(f"🔄 REV: Đóng {old_pos['dir']} @ {current_price:.2f} | Qty: {old_pos['qty']:.4f}")
                                except Exception as e:
                                    logger.error(f"❌ REV Close Error: {e}")

                            # BƯỚC 2: Mở lệnh mới ngược chiều
                            await asyncio.to_thread(
                                exchange.create_market_order, BYBIT_SYMBOL,
                                side_str, target_qty, params=params
                            )
                            logger.info(f"✅ REV: Mở {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

                    elif existing_same:
                        logger.info(f"ℹ️ Đã có lệnh {b_dir} cùng chiều → Bỏ qua")
                    else:
                        if len(state_tracker.bybit_positions) < MAX_BYBIT_POSITIONS:
                            await asyncio.to_thread(exchange.create_market_order, BYBIT_SYMBOL, side_str, target_qty, params=params)
                            logger.info(f"✅ Bybit Mở mới {b_dir} @ {current_price:.2f} | Qty: {target_qty:.4f}")

        except Exception as e:
            logger.error(f"❌ Bybit Error: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    bot = Bot(token=TELEGRAM_TOKEN) if (ENABLE_TELEGRAM and TELEGRAM_TOKEN) else None
    exchange = None
    if ENABLE_BYBIT_TRADING and BYBIT_API_KEY:
        exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_API_SECRET,
            'options': {'defaultType': 'linear'}
        })

    features = []
    if ENABLE_BYBIT_TRADING: features.append("📊 Bybit Auto-Trade (ETHUSDT)")
    if ENABLE_TELEGRAM:      features.append("📱 Telegram")
    features_str = "\n".join([f"   • {f}" for f in features]) if features else "   • Analysis Only"

    startup_msg = f"""
🚀 <b>BOT ETHUSDT KHỞI ĐỘNG (V3 Alts)</b>

⏱ Timeframe: M15 + H1
🔧 Features:
{features_str}

📌 Config:
   Kelly: STRONG={KELLY_FRACTIONS['STRONG_TREND']*100:.0f}% | TREND={KELLY_FRACTIONS['TREND']*100:.0f}% | WEAK={KELLY_FRACTIONS['WEAK_TREND']*100:.0f}%

⏰ {get_vn_time()}
"""
    await send_telegram(bot, startup_msg)

    if exchange:
        for _ in range(3):
            try:
                await asyncio.to_thread(exchange.set_position_mode, True, BYBIT_SYMBOL)
                await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                break
            except Exception as e:
                if "110025" in str(e):
                    try: await asyncio.to_thread(exchange.set_leverage, LEVERAGE, BYBIT_SYMBOL)
                    except: pass
                    break
                await asyncio.sleep(1)
        try:
            bal = await asyncio.to_thread(exchange.fetch_balance)
            equity = float(bal['USDT']['total'])
            logger.info(f"💰 Startup Balance: {equity:.2f} USDT")
        except: pass

    while True:
        try:
            now  = datetime.now()
            wait = (15 - (now.minute % 15)) * 60 - now.second + 3
            if wait < 10: wait = 1
            logger.info(f"⏳ Waiting {wait}s for next M15 candle...")
            await asyncio.sleep(wait)
            await run_scan(bot, exchange)
        except Exception as e:
            logger.error(f"❌ Main Loop Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
