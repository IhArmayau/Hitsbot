import asyncio
import aiohttp
import pandas as pd
import ta
import logging
import os
from datetime import datetime, timezone
from flask import Flask
import threading
import time
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    "pools": [
        {"symbol": "ETH/USDT", "network": "eth", "pool_address": os.environ.get("ETH_POOL_ADDRESS")},
        {"symbol": "INJ/USDT", "network": "arbitrum", "pool_address": os.environ.get("INJ_POOL_ADDRESS")},
        {"symbol": "ADA/USDT", "network": "bsc", "pool_address": os.environ.get("ADA_POOL_ADDRESS")},
        {"symbol": "XPL/USDT", "network": "eth", "pool_address": os.environ.get("XPL_POOL_ADDRESS")},
        {"symbol": "XRP/USDT", "network": "eth", "pool_address": os.environ.get("XRP_POOL_ADDRESS")},
    ],
    "timeframes": {
        "entry": "15m",
        "confirmation_1h": "1h",
        "confirmation_4h": "4h"
    },
    "indicators": {
        "ema_short": 9, "ema_long": 21, "ema_trend": 50,
        "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
        "adx_period": 14, "adx_threshold": 25,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_window": 20, "bb_std": 2,
        "atr_period": 14, "atr_sl": 1.5, "atr_tp": 3
    },
    "min_confidence": 70,
    "top_signals": 2,
    "telegram": {
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID")
    },
    "csv_file": "signals.csv",
    "poll_interval_sec": 900,    # 15 minutes
    "status_interval_sec": 600,  # 10 minutes
    "cache_ttl_sec": 300,
    "rate_limit": 5,
    "max_failures": 3,
    "skip_duration": 300,
    "max_rows": 1000
}

# -----------------------------
# LOGGING & GLOBALS
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
signals_memory = pd.DataFrame(columns=[
    'time','symbol','signal','confidence','strength','entry','sl','tp','candle_time'
])
ohlcv_cache = {}
semaphore = asyncio.Semaphore(CONFIG['rate_limit'])
pool_failures = {}
pool_skip_until = {}

# -----------------------------
# FLASK UPTIME SERVER
# -----------------------------
app = Flask(__name__)
@app.route("/")
def home():
    return "DEX Perpetual Signal Bot is running! 🤖"

def run_flask():
    logging.info("✅ Flask web server running on port 10000...")
    app.run(host="0.0.0.0", port=10000)

# -----------------------------
# ENV CHECK
# -----------------------------
def check_env():
    logging.info("🔍 Checking environment variables...")
    for var in ["ETH_POOL_ADDRESS", "INJ_POOL_ADDRESS", "ADA_POOL_ADDRESS", "XPL_POOL_ADDRESS", "XRP_POOL_ADDRESS", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        val = os.environ.get(var)
        if val:
            logging.info(f"{var} = SET ({str(val)[:8]}...)")
        else:
            logging.warning(f"{var} = NOT SET ❌")

# -----------------------------
# UTILITIES
# -----------------------------
def load_signals():
    global signals_memory
    if not os.path.exists(CONFIG['csv_file']):
        logging.info("No existing signals.csv found — starting fresh.")
        return
    try:
        df = pd.read_csv(CONFIG['csv_file'])
        df['time'] = pd.to_datetime(df['time'], errors='coerce').fillna(pd.Timestamp.now(tz=timezone.utc))
        df['candle_time'] = pd.to_datetime(df['candle_time'], errors='coerce').fillna(df['time'])
        signals_memory = df[['time','symbol','signal','confidence','strength','entry','sl','tp','candle_time']].copy()
        logging.info(f"📄 Loaded {len(signals_memory)} historical signals from CSV")
    except Exception as e:
        logging.warning(f"⚠️ Failed to load CSV (safe fallback): {e}")
        signals_memory = pd.DataFrame(columns=[
            'time','symbol','signal','confidence','strength','entry','sl','tp','candle_time'
        ])

def save_signal(signal):
    global signals_memory
    try:
        s = signal.copy()
        if isinstance(s.get('time'), (pd.Timestamp, datetime)):
            s['time'] = pd.to_datetime(s['time']).isoformat()
        if isinstance(s.get('candle_time'), (pd.Timestamp, datetime)):
            s['candle_time'] = pd.to_datetime(s['candle_time']).isoformat()
        signals_memory = pd.concat([signals_memory, pd.DataFrame([s])], ignore_index=True)
        if len(signals_memory) > CONFIG['max_rows']:
            signals_memory = signals_memory.iloc[-CONFIG['max_rows']:]
        signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')
        logging.info(f"💾 Saved signal to CSV: {s.get('symbol')} {s.get('signal')}")
    except Exception as e:
        logging.error(f"❌ Failed to save signal: {e}")

def is_duplicate(symbol, signal, candle_time):
    try:
        candle_time = pd.to_datetime(candle_time).replace(second=0, microsecond=0)
    except Exception:
        candle_time = pd.to_datetime(candle_time, errors='coerce')
    dup = not signals_memory[
        (signals_memory['symbol']==symbol) &
        (signals_memory['signal']==signal) &
        (pd.to_datetime(signals_memory['candle_time'], errors='coerce') == candle_time)
    ].empty
    if dup:
        logging.info(f"⚠️ Duplicate detected for {symbol} {signal} at {candle_time}")
    return dup

# -----------------------------
# TELEGRAM
# -----------------------------
async def send_telegram(session, message):
    if not CONFIG['telegram']['bot_token'] or not CONFIG['telegram']['chat_id']:
        logging.warning("❌ Telegram not configured, skipping message.")
        return
    url = f"https://api.telegram.org/bot{CONFIG['telegram']['bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG['telegram']['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            if resp.status == 200:
                logging.info("✅ Telegram message sent successfully.")
            else:
                text = await resp.text()
                logging.warning(f"⚠️ Telegram send failed: {resp.status} - {text}")
    except Exception as e:
        logging.error(f"❌ Telegram exception: {e}")

# -----------------------------
# FETCH OHLCV (GeckoTerminal)
# -----------------------------
async def fetch_ohlcv(session, network, pool_address, timeframe):
    now = time.time()
    if not pool_address:
        logging.warning(f"❌ Missing pool address for {network}. Skipping.")
        return pd.DataFrame()

    key = (network, pool_address, timeframe)
    async with semaphore:
        url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logging.warning(f"⚠️ GeckoTerminal {resp.status} for {pool_address}")
                    return pd.DataFrame()
                data = await resp.json()
                ohlcv_list = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
                if not ohlcv_list:
                    return pd.DataFrame()
                df = pd.DataFrame(ohlcv_list, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
                df.set_index("timestamp", inplace=True)
                df = df.dropna().sort_index()
                ohlcv_cache[key] = (now, df.copy())
                return df
        except Exception as e:
            logging.error(f"❌ OHLCV fetch error: {e}")
            return pd.DataFrame()

# -----------------------------
# INDICATORS & SIGNAL LOGIC
# -----------------------------
def add_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    ind = CONFIG["indicators"]
    try:
        df = df.copy()
        df["ema_short"] = df["close"].ewm(span=ind["ema_short"], adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=ind["ema_long"], adjust=False).mean()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], ind["rsi_period"]).rsi()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ind["adx_period"]).adx()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ind["atr_period"]).average_true_range()
        return df.dropna()
    except Exception as e:
        logging.error(f"❌ Indicator error: {e}")
        return pd.DataFrame()

def compute_signal(latest, prev):
    buy = sell = 0
    try:
        if prev["ema_short"] < prev["ema_long"] and latest["ema_short"] > latest["ema_long"]:
            buy += 20
        if prev["ema_short"] > prev["ema_long"] and latest["ema_short"] < latest["ema_long"]:
            sell += 20
        if latest["rsi"] < CONFIG["indicators"]["rsi_oversold"]:
            buy += 20
        if latest["rsi"] > CONFIG["indicators"]["rsi_overbought"]:
            sell += 20
    except Exception as e:
        logging.error(f"Signal calc error: {e}")
    return buy, sell

# -----------------------------
# COLLECT SIGNALS
# -----------------------------
async def collect_signals(session):
    signals = []
    for pool in CONFIG["pools"]:
        df_entry = add_indicators(await fetch_ohlcv(session, pool["network"], pool["pool_address"], CONFIG["timeframes"]["entry"]))
        if df_entry.empty or len(df_entry) < 2:
            continue
        latest, prev = df_entry.iloc[-1], df_entry.iloc[-2]
        buy, sell = compute_signal(latest, prev)
        confidence = max(buy, sell)
        if confidence < CONFIG["min_confidence"]:
            continue
        entry_signal = "BUY" if buy > sell else "SELL"
        candle_time = latest.name
        if is_duplicate(pool["symbol"], entry_signal, candle_time):
            continue
        entry_price = latest["close"]
        atr = latest.get("atr", None)
        if atr is None or pd.isna(atr):
            sl = entry_price * (0.995 if entry_signal == "BUY" else 1.005)
            tp = entry_price * (1.01 if entry_signal == "BUY" else 0.99)
        else:
            sl = entry_price - atr * CONFIG["indicators"]["atr_sl"] if entry_signal == "BUY" else entry_price + atr * CONFIG["indicators"]["atr_sl"]
            tp = entry_price + atr * CONFIG["indicators"]["atr_tp"] if entry_signal == "BUY" else entry_price - atr * CONFIG["indicators"]["atr_tp"]
        signals.append({
            "time": datetime.now(timezone.utc),
            "symbol": pool["symbol"],
            "signal": entry_signal,
            "confidence": confidence,
            "strength": confidence,
            "entry": float(entry_price),
            "sl": float(sl),
            "tp": float(tp),
            "candle_time": candle_time
        })
    return sorted(signals, key=lambda x: x["confidence"], reverse=True)[:CONFIG["top_signals"]]

# -----------------------------
# MAIN BOT
# -----------------------------
async def run_bot(session):
    load_signals()
    while True:
        signals = await collect_signals(session)
        for s in signals:
            msg = f"*{s['symbol']}* | *{s['signal']}* | Entry: {s['entry']:.6f} | SL: {s['sl']:.6f} | TP: {s['tp']:.6f} | Confidence: {s['confidence']}%"
            await send_telegram(session, msg)
            save_signal(s)
        await asyncio.sleep(CONFIG["poll_interval_sec"])

async def main():
    check_env()
    async with aiohttp.ClientSession() as session:
        await run_bot(session)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(1)
    logging.info("🚀 Starting DEX Signal Bot for ETH, INJ, ADA, XPL, XRP ...")
    asyncio.run(main())
