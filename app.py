import time
import requests
import hmac
import hashlib
import json
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
from threading import Thread
from collections import deque
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ====== Flask ======
app = Flask(__name__)

@app.route('/healthz')
def healthz():
    return "ok", 200

def log_status(title, value, color="white"):
    print(f"{title:<25}: {value}")

# ===== Trade tracking variables =====
total_trades = 0
successful_trades = 0
failed_trades = 0
trade_log = deque(maxlen=40)
compound_profit = 0.0
last_direction = None
daily_loss = 0.0
current_day = None
# ===================================

# Cooldown period to avoid duplicate signals (10 minutes)
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_SECONDS", "600"))
last_trade_time = 0

# ========== Trading Configuration ==========
API_KEY = os.getenv("BINGX_API_KEY")
API_SECRET = os.getenv("BINGX_API_SECRET")
BASE_URL = "https://open-api.bingx.com"

SYMBOL = os.getenv("SYMBOL", "DOGE-USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = float(os.getenv("LEVERAGE", "10"))
TRADE_PORTION = float(os.getenv("TRADE_PORTION", "0.60"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
TOLERANCE = float(os.getenv("TOLERANCE", "0.0005"))

# Risk management parameters
MIN_ATR = float(os.getenv("MIN_ATR", "0.001"))
MIN_TP_PERCENT = float(os.getenv("MIN_TP_PERCENT", "0.75"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "50"))  # USDT stop for the day
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "20"))
# ===========================================

# Trading state variables
position_open = False
position_side = None
entry_price = 0.0
tp_price = 0.0
sl_price = 0.0
current_quantity = 0.0
current_atr = 0.0
current_pnl = 0.0
current_price = 0.0
ema_200_value = 0.0
rsi_value = 0.0
adx_value = 0.0
update_time = ""
trail_active = False

# Compound profit variables
initial_balance = 0.0

# ===== BingX core (unchanged) =====
def get_signature(params):
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def safe_api_request(method, endpoint, params=None, data=None):
    try:
        url = f"{BASE_URL}{endpoint}"
        headers = {"X-BX-APIKEY": API_KEY}
        timestamp = str(int(time.time() * 1000))
        if params is None:
            params = {}
        params["timestamp"] = timestamp
        params["signature"] = get_signature(params)
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=15)
        elif method == "POST":
            response = requests.post(url, headers=headers, params=params, json=data, timeout=15)
        else:
            return None
        if response.status_code != 200:
            print(f"❌ API request failed {response.status_code}: {response.text}")
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            print(f"❌ Failed to parse JSON response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return None

def get_balance():
    try:
        timestamp = str(int(time.time() * 1000))
        params = {"timestamp": timestamp}
        signature = get_signature(params)
        url = f"{BASE_URL}/openApi/swap/v2/user/balance?timestamp={timestamp}&signature={signature}"
        headers = {"X-BX-APIKEY": API_KEY}
        response = requests.get(url, headers=headers, timeout=15)
        result = response.json()
        if isinstance(result, dict) and result.get("code") == 0:
            balance_data = result.get("data", {})
            if isinstance(balance_data.get("balance"), list):
                for asset in balance_data["balance"]:
                    if asset.get("asset") == "USDT":
                        return float(asset.get("availableBalance", 0.0))
            elif isinstance(balance_data.get("balance"), dict):
                asset = balance_data["balance"]
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableMargin", 0.0))
            print("❌ USDT balance not found in response")
        else:
            print(f"❌ Balance request failed: {result}")
    except Exception as e:
        print(f"❌ Error fetching balance: {str(e)}")
    return 0.0

def get_open_position():
    try:
        params = {"symbol": SYMBOL}
        response = safe_api_request("GET", "/openApi/swap/v2/user/positions", params)
        if response and isinstance(response, dict) and "data" in response:
            for position in response["data"]:
                if (isinstance(position, dict) and 
                    "entryPrice" in position and 
                    "positionAmt" in position and
                    float(position.get("positionAmt", 0)) != 0):
                    return {
                        "side": "BUY" if float(position["positionAmt"]) > 0 else "SELL",
                        "entryPrice": float(position["entryPrice"]),
                        "positionAmt": abs(float(position["positionAmt"])),
                        "unrealizedProfit": float(position.get("unrealizedProfit", 0))
                    }
        return None
    except Exception as e:
        print(f"❌ Error in get_open_position: {e}")
        return None

def get_klines():
    try:
        response = requests.get(
            f"{BASE_URL}/openApi/swap/v2/quote/klines",
            params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 300},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching klines: {e}")
        return pd.DataFrame()
# ===== End BingX core =====

def calculate_adx(df, period=14):
    try:
        if len(df) < period * 2:
            return pd.Series()
        high = df["high"]; low = df["low"]; close = df["close"]
        plus_dm = high.diff(); minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
        tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period).mean()
        return adx
    except Exception as e:
        print(f"❌ Error calculating ADX: {e}")
        return pd.Series()

def calculate_ema(series, period):
    if len(series) < period:
        return pd.Series()
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    if len(series) < period:
        return pd.Series()
    return series.rolling(period).mean()

def price_range_percent(df, lookback=20):
    if len(df) < lookback:
        return 0.0
    recent = df["close"].iloc[-lookback:]
    highest = recent.max(); lowest = recent.min()
    return ((highest - lowest) / max(lowest, 1e-9)) * 100

def calculate_supertrend(df, period=10, multiplier=3):
    try:
        if len(df) < period * 2:
            return pd.Series(), pd.Series()
        high = df["high"]; low = df["low"]; close = df["close"]
        hl2 = (high + low) / 2
        atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=period)
        atr = atr_indicator.average_true_range()
        if atr.empty:
            return pd.Series(), pd.Series()
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        direction = pd.Series(np.ones(len(close)), index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
        supertrend = np.where(direction == 1, lower_band, upper_band)
        return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)
    except Exception as e:
        print(f"❌ Error calculating Supertrend: {e}")
        return pd.Series(), pd.Series()

def calculate_tp_sl(entry_price, atr_value, direction):
    if direction == "BUY":
        tp = entry_price + atr_value * 1.2
        sl = entry_price - atr_value * 0.8
    else:
        tp = entry_price - atr_value * 1.2
        sl = entry_price + atr_value * 0.8
    return round(tp, 5), round(sl, 5)

def create_tp_sl_orders():
    global position_open
    if not position_open or current_quantity <= 0 or entry_price <= 0:
        print("⚠️ Skipping TP/SL creation — Missing data!"); return False
    time.sleep(1)
    if position_side == "BUY":
        tp_side = "SELL"; sl_side = "SELL"
    else:
        tp_side = "BUY"; sl_side = "BUY"
    tp_params = {"symbol": SYMBOL,"side": tp_side,"positionSide":"BOTH","type":"TAKE_PROFIT_MARKET",
                 "quantity": current_quantity,"stopPrice": f"{tp_price:.5f}","workingType":"MARK_PRICE"}
    sl_params = {"symbol": SYMBOL,"side": sl_side,"positionSide":"BOTH","type":"STOP_MARKET",
                 "quantity": current_quantity,"stopPrice": f"{sl_price:.5f}","workingType":"MARK_PRICE"}
    try:
        tp_response = safe_api_request("POST","/openApi/swap/v2/trade/order",params=tp_params)
        if tp_response and tp_response.get("code")==0:
            print(f"✅ TP order placed @ {tp_price:.5f}")
        else:
            print(f"❌ Failed to place TP order: {tp_response}"); close_position("NO_TP", current_price); return False
        sl_response = safe_api_request("POST","/openApi/swap/v2/trade/order",params=sl_params)
        if sl_response and sl_response.get("code")==0:
            print(f"✅ SL order placed @ {sl_price:.5f}"); return True
        else:
            print(f"❌ Failed to place SL order: {sl_response}"); close_position("NO_SL", current_price); return False
    except Exception as e:
        print(f"❌ Error creating TP/SL orders: {e}"); close_position("ERROR", current_price); return False

def place_order(side, quantity):
    global position_open, position_side, entry_price, current_quantity, tp_price, sl_price, last_trade_time, trail_active
    now = time.time()
    if now - last_trade_time < COOLDOWN_PERIOD:
        print(f"⏳ Cooldown: {int(COOLDOWN_PERIOD - (now-last_trade_time))}s"); return False
    if position_open:
        print("🚫 Position already open"); return False
    try:
        atr = max(current_atr, MIN_ATR)
        estimated_entry = current_price
        estimated_tp, _ = calculate_tp_sl(estimated_entry, atr, side)
        tp_distance = abs(estimated_tp - estimated_entry)
        if estimated_entry <= 0:
            print("❌ Invalid price"); return False
        tp_percent = (tp_distance / estimated_entry) * 100
        if tp_percent < MIN_TP_PERCENT:
            print(f"🚫 TP too small: {tp_percent:.3f}%"); return False
        if adx_value < 20:
            print("🚫 ADX too weak"); return False
        params = {"symbol": SYMBOL,"side": side,"positionSide":"BOTH","type":"MARKET","quantity": quantity}
        response = safe_api_request("POST","/openApi/swap/v2/trade/order",params=params)
        if response and response.get("code")==0:
            order_data = response["data"]
            entry = float(order_data.get("avgPrice") or estimated_entry)
            entry_price = entry; position_side = side; current_quantity = quantity; position_open = True
            tp_price, sl_price = calculate_tp_sl(entry_price, atr, position_side)
            last_trade_time = now; trail_active = False
            print(f"\n{'🟢 BUY' if side=='BUY' else '🔴 SELL'} @ {entry_price:.5f}")
            print(f"🎯 TP: {tp_price:.5f} | 🛑 SL: {sl_price:.5f} | ATR: {atr:.5f}")
            if not create_tp_sl_orders():
                print("🛑 Trade aborted - protection orders failed"); position_open=False; return False
            return True
        else:
            print(f"❌ Failed to place {side} order: {response}")
            return False
    except Exception as e:
        print(f"❌ Error placing order: {e}")
        return False

def close_position(reason, exit_price):
    global position_open, position_side, entry_price, current_quantity, tp_price, sl_price
    global total_trades, successful_trades, failed_trades, compound_profit, last_trade_time, last_direction, daily_loss
    if not position_open or position_side is None:
        print("⚠️ No open position to close"); return False
    close_side = "SELL" if position_side=="BUY" else "BUY"
    params = {"symbol": SYMBOL,"side": close_side,"positionSide":"BOTH","type":"MARKET","quantity": current_quantity}
    try:
        response = safe_api_request("POST","/openApi/swap/v2/trade/order",params=params)
        if response and response.get("code")==0:
            order_data = response["data"]
            exit_p = float(order_data.get("avgPrice") or current_price)
            if position_side=="BUY":
                profit = (exit_p - entry_price) * current_quantity
                profit_pct = ((exit_p - entry_price) / entry_price) * 100
            else:
                profit = (entry_price - exit_p) * current_quantity
                profit_pct = ((entry_price - exit_p) / entry_price) * 100
            compound_profit += profit; total_trades += 1
            if reason=="TP": successful_trades += 1
            else: failed_trades += 1; daily_loss += max(-profit, 0)
            trade_log.appendleft({'side':position_side,'entry_price':entry_price,'exit_price':exit_p,'result':reason,'profit':profit,'time':time.strftime("%Y-%m-%d %H:%M:%S")})
            last_direction = position_side; last_trade_time = time.time()
            print(f"\n💼 Closed {position_side} @ {exit_p:.5f} | Entry: {entry_price:.5f}")
            print(f"📈 Profit: {profit:.4f} USDT | 📊 Change: {profit_pct:.2f}% | 💰 Total: {compound_profit:.4f}")
            position_open=False; position_side=None; entry_price=0.0; current_quantity=0.0; tp_price=0.0; sl_price=0.0
            print("🔄 Waiting 5s for balance update..."); time.sleep(5)
            return True
        else:
            print(f"❌ Failed to close position: {response}"); return False
    except Exception as e:
        print(f"❌ Error closing position: {e}"); return False

def check_position_status():
    global current_price, current_pnl, sl_price, trail_active
    if not position_open or position_side is None: return
    if position_side == "BUY":
        current_pnl = (current_price - entry_price) * current_quantity
    else:
        current_pnl = (entry_price - current_price) * current_quantity

    # --- ATR-based trailing & breakeven ---
    atr = max(current_atr, MIN_ATR)
    if position_side=="BUY":
        if current_price - entry_price >= atr and not trail_active:
            sl_price = round(entry_price,5)  # move to breakeven
            trail_active = True
            print(f"🟩 Breakeven set at {sl_price:.5f}")
        if trail_active:
            new_sl = round(current_price - atr,5)
            if new_sl > sl_price:
                sl_price = new_sl
    else:
        if entry_price - current_price >= atr and not trail_active:
            sl_price = round(entry_price,5)
            trail_active = True
            print(f"🟩 Breakeven set at {sl_price:.5f}")
        if trail_active:
            new_sl = round(current_price + atr,5)
            if new_sl < sl_price:
                sl_price = new_sl

    # Close conditions
    if position_side == "BUY":
        if current_price >= tp_price - TOLERANCE:
            close_position("TP", current_price)
        elif current_price <= sl_price + TOLERANCE:
            close_position("SL", current_price)
    else:
        if current_price <= tp_price + TOLERANCE:
            close_position("TP", current_price)
        elif current_price >= sl_price - TOLERANCE:
            close_position("SL", current_price)

def resume_open_position():
    global position_open, position_side, entry_price, current_quantity, current_atr, tp_price, sl_price
    try:
        position = get_open_position()
        if position:
            position_side = position["side"]; entry_price = position["entryPrice"]
            current_quantity = position["positionAmt"]; position_open = True
            atr = max(current_atr, MIN_ATR)
            tp_price, sl_price = calculate_tp_sl(entry_price, atr, position_side)
            print(f"\n▶️ RESUMING OPEN POSITION: {position_side} @ {entry_price:.5f} qty={current_quantity}")
            create_tp_sl_orders()
            return True
        return False
    except Exception as e:
        print(f"❌ Error resuming position: {e}")
        return False

# ====== Dashboard route ======
@app.route('/')
def dashboard():
    return render_template_string('''
    <!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>DOGE Trading Bot</title>
    <style>
    body{background:#0b1220;color:#e5e7eb;font-family:system-ui,Segoe UI,Arial;margin:0;padding:24px}
    .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));}
    .card{background:#111827;border:1px solid #243244;border-radius:12px;padding:16px}
    .title{color:#9ca3af;margin-bottom:8px;font-weight:600}
    .big{font-size:24px;font-weight:700}
    .pos{color:#10b981}.neg{color:#ef4444}
    .log{max-height:420px;overflow:auto}
    .row{display:flex;justify-content:space-between;border-bottom:1px solid #243244;padding:8px 0}
    </style></head><body>
    <h2>🪙 DOGE/USDT Trading Bot — Render</h2>
    <div class="grid">
      <div class="card">
        <div class="title">📊 Performance</div>
        <div class="row"><span>Total Trades</span><span class="big">{{ total_trades }}</span></div>
        <div class="row"><span>Wins</span><span class="big pos">{{ successful_trades }}</span></div>
        <div class="row"><span>Losses</span><span class="big neg">{{ failed_trades }}</span></div>
        <div class="row"><span>Total Profit (USDT)</span><span class="big {% if compound_profit>=0 %}pos{% else %}neg{% endif %}">{{ compound_profit|round(4) }}</span></div>
        <div class="row"><span>Daily Loss</span><span class="big {% if daily_loss<=0 %}pos{% else %}neg{% endif %}">{{ daily_loss|round(2) }}</span></div>
      </div>
      <div class="card">
        <div class="title">📈 Market</div>
        <div class="row"><span>Price</span><span>{{ current_price|round(5) }}</span></div>
        <div class="row"><span>EMA 200</span><span>{{ ema_200|round(5) }}</span></div>
        <div class="row"><span>RSI</span><span>{{ rsi|round(1) }}</span></div>
        <div class="row"><span>ADX</span><span>{{ adx|round(1) }}</span></div>
        <div class="row"><span>ATR</span><span>{{ atr|round(5) }}</span></div>
        <div class="row"><span>Update</span><span>{{ update_time }}</span></div>
      </div>
      <div class="card">
        <div class="title">⚙️ Status</div>
        {% if position_open %}
          <div class="row"><span>Position</span><span>ACTIVE ({{ position_side }})</span></div>
          <div class="row"><span>Entry</span><span>{{ position_entry|round(5) }}</span></div>
          <div class="row"><span>TP / SL</span><span>{{ position_tp|round(5) }} / {{ position_sl|round(5) }}</span></div>
          <div class="row"><span>PnL</span><span>{{ position_pnl|round(4) }} USDT</span></div>
        {% else %}
          <div class="row"><span>Position</span><span>NONE</span></div>
          <div class="row"><span>Signal</span><span>{{ signal }}</span></div>
        {% endif %}
      </div>
    </div>
    <div class="card log">
      <div class="title">📜 Recent Trades</div>
      {% for t in trade_log %}
        <div class="row"><span>{{ t.time }} — {{ t.side }} {{ t.entry_price|round(5) }} → {{ t.exit_price|round(5) }} ({{ t.result }})</span><span class="{% if t.profit>=0 %}pos{% else %}neg{% endif %}">{{ t.profit|round(4) }} USDT</span></div>
      {% endfor %}
    </div>
    </body></html>
    ''',
    total_trades=total_trades,
    successful_trades=successful_trades,
    failed_trades=failed_trades,
    compound_profit=compound_profit,
    daily_loss=daily_loss,
    position_open=position_open,
    position_side=position_side,
    position_entry=entry_price,
    position_tp=tp_price,
    position_sl=sl_price,
    position_pnl=current_pnl,
    current_price=current_price,
    ema_200=ema_200_value,
    rsi=rsi_value,
    adx=adx_value,
    atr=current_atr,
    update_time=update_time,
    signal="BUY" if last_direction=="SELL" else "SELL" if last_direction=="BUY" else "WAIT")

# ====== Main Loop ======
def main_bot_loop():
    global current_atr, current_price, ema_200_value, rsi_value, adx_value, update_time
    global current_day, daily_loss

    print("🚀 Starting DOGE Trading Bot...")
    print(f"Symbol={SYMBOL} | Leverage={LEVERAGE}x | Risk/Trade={TRADE_PORTION*100:.0f}% | Interval={INTERVAL}")

    initial_balance = get_balance()
    if initial_balance <= 0:
        print("⚠️ No balance or API not ready — starting in IDLE mode.")
        initial_balance = 0.0

    df = get_klines()
    if not df.empty:
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_PERIOD)
        atr_series = atr_indicator.average_true_range()
        current_atr = atr_series.iloc[-1] if not atr_series.empty else MIN_ATR

    resume_open_position()

    while True:
        try:
            update_time = time.strftime("%Y-%m-%d %H:%M:%S")
            # reset daily loss counter on new UTC day
            day = time.strftime("%Y-%m-%d")
            if current_day != day:
                current_day = day
                daily_loss = 0.0

            sleep_time = 15 if position_open else 60
            df = get_klines()
            if df.empty:
                print("❌ Failed to get market data, retrying..."); time.sleep(sleep_time); continue
            if len(df) < 60:
                print(f"⚠️ Insufficient data ({len(df)} candles), waiting..."); time.sleep(sleep_time); continue

            close_prices = df["close"]
            current_price = close_prices.iloc[-1]
            atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_PERIOD)
            atr_series = atr_indicator.average_true_range()
            current_atr = atr_series.iloc[-1] if not atr_series.empty else MIN_ATR

            rsi_series = RSIIndicator(close=df["close"], window=14).rsi()
            ema_20 = calculate_ema(close_prices, 20)
            ema_50 = calculate_ema(close_prices, 50)
            ema_200 = calculate_ema(close_prices, 200)
            adx_series = calculate_adx(df)
            supertrend_line, supertrend_trend = calculate_supertrend(df)

            rsi_value = rsi_series.iloc[-2] if len(rsi_series)>=2 else 0
            ema_200_value = ema_200.iloc[-1] if not ema_200.empty else 0
            adx_value = adx_series.iloc[-1] if not adx_series.empty else 0
            current_supertrend = supertrend_trend.iloc[-1] if not supertrend_trend.empty else 0

            # Volatility & spike filter
            current_close = close_prices.iloc[-1]
            previous_close = close_prices.iloc[-2]
            spike = abs(current_close - previous_close) > current_atr * 1.8
            price_range = price_range_percent(df)

            # ===== Enhanced 15m strategy =====
            # Trend filter + momentum + supertrend + pullback re-entry
            pullback_ok_buy = close_prices.iloc[-2] <= ema_20.iloc[-2] and current_close > ema_20.iloc[-1]
            pullback_ok_sell = close_prices.iloc[-2] >= ema_20.iloc[-2] and current_close < ema_20.iloc[-1]

            long_ok = (current_price > ema_200_value and
                       ema_20.iloc[-1] > ema_50.iloc[-1] and
                       rsi_value >= 52 and rsi_value <= 72 and
                       adx_value >= 25 and
                       current_supertrend > 0 and
                       pullback_ok_buy)

            short_ok = (current_price < ema_200_value and
                        ema_20.iloc[-1] < ema_50.iloc[-1] and
                        rsi_value <= 48 and rsi_value >= 28 and
                        adx_value >= 25 and
                        current_supertrend < 0 and
                        pullback_ok_sell)

            # ===== Risk controls =====
            if daily_loss >= MAX_DAILY_LOSS:
                print(f"🛑 Daily loss limit reached ({daily_loss:.2f} USDT) — pausing until next day.")
                check_position_status(); time.sleep(90); continue

            current_balance = get_balance()
            total_balance = initial_balance + compound_profit
            trade_usdt = min(total_balance * TRADE_PORTION, current_balance)
            effective_usdt = trade_usdt * LEVERAGE
            quantity = round(effective_usdt / max(current_price, 1e-9), 2)

            print(f"Price={current_price:.5f} | ATR={current_atr:.5f} | RSI={rsi_value:.1f} | ADX={adx_value:.1f} | Bal={current_balance:.2f}")
            check_position_status()

            if not position_open:
                now = time.time()
                if now - last_trade_time < COOLDOWN_PERIOD:
                    print(f"🕒 Cooldown active ({int(COOLDOWN_PERIOD - (now-last_trade_time))}s)")
                elif spike:
                    print(f"⛔ Spike candle detected — skip")
                elif price_range <= 1.0:
                    print(f"⛔ Price range too low ({price_range:.2f}%) — skip")
                else:
                    if long_ok:
                        place_order("BUY", quantity)
                    elif short_ok:
                        place_order("SELL", quantity)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(60)

def keep_alive():
    url = os.getenv("KEEPALIVE_URL")
    if not url: return
    def ping():
        while True:
            try:
                requests.get(url, timeout=5)
                print("🟢 Keep-alive ping sent")
            except:
                print("🔴 Keep-alive failed")
            time.sleep(300)
    t = Thread(target=ping, daemon=True); t.start()

keep_alive()

if __name__ == '__main__':
    bot_thread = Thread(target=main_bot_loop, daemon=True)
    bot_thread.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))