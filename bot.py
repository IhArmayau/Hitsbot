import asyncio
import aiohttp
import pandas as pd
import ta
import logging
import os
from datetime import datetime, timezone
from flask import Flask
from dotenv import load_dotenv
from threading import Thread

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
    "timeframe": os.environ.get("TIMEFRAME", "hour"),  # Allowed: day, hour, minute, second
    "telegram": {
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID")
    },
    "csv_file": "signals.csv",
    "poll_interval_sec": 900  # 15 minutes
}

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -----------------------------
# GLOBALS
# -----------------------------
signals_memory = pd.DataFrame(columns=[
    'time','symbol','signal','entry','sl','tp','candle_time'
])
ohlcv_cache = {}

# -----------------------------
# FLASK SERVER FOR UPTIME
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "DEX Signal Bot is running!"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# -----------------------------
# TELEGRAM
# -----------------------------
async def send_telegram(session, message):
    token = CONFIG["telegram"]["bot_token"]
    chat_id = CONFIG["telegram"]["chat_id"]
    if not token or not chat_id:
        logger.warning("Telegram not configured, skipping message")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                logger.info("Telegram message sent ✅")
            else:
                text = await resp.text()
                logger.warning(f"Telegram send failed: {resp.status} {text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# -----------------------------
# LOAD & SAVE SIGNALS
# -----------------------------
def load_signals():
    global signals_memory
    if os.path.exists(CONFIG["csv_file"]):
        try:
            df = pd.read_csv(CONFIG["csv_file"])
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
            df["candle_time"] = pd.to_datetime(df["candle_time"], errors='coerce')
            signals_memory = df
            logger.info(f"Loaded {len(df)} signals from CSV")
        except Exception as e:
            logger.warning(f"Failed to load CSV: {e}")

def save_signal(signal):
    global signals_memory
    signals_memory = pd.concat([signals_memory, pd.DataFrame([signal])], ignore_index=True)
    signals_memory.to_csv(CONFIG["csv_file"], index=False)
    logger.info(f"Saved signal: {signal['symbol']} {signal['signal']}")

# -----------------------------
# FETCH OHLCV FROM GeckoTerminal
# -----------------------------
async def fetch_ohlcv(session, network, pool_address, timeframe):
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status != 200:
                logger.warning(f"GeckoTerminal {resp.status} for {pool_address}")
                return pd.DataFrame()
            data = await resp.json()
            ohlcv_list = data.get("data", {}).get("items", [])
            if not ohlcv_list:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv_list)
            df["timestamp"] = pd.to_datetime(df["time"], unit='s')
            df.set_index("timestamp", inplace=True)
            df = df[["open","high","low","close","volume"]].astype(float)
            return df
    except Exception as e:
        logger.error(f"OHLCV fetch error: {e}")
        return pd.DataFrame()

# -----------------------------
# INDICATORS & SIGNALS
# -----------------------------
def compute_indicators(df):
    if df.empty:
        return df
    df["ema_short"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=21, adjust=False).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    return df.dropna()

def generate_signal(df, symbol):
    if df.empty or len(df) < 2:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signal = None
    if prev["ema_short"] < prev["ema_long"] and latest["ema_short"] > latest["ema_long"] and latest["rsi"] < 60:
        signal = "BUY"
    elif prev["ema_short"] > prev["ema_long"] and latest["ema_short"] < latest["ema_long"] and latest["rsi"] > 40:
        signal = "SELL"
    return signal

# -----------------------------
# MAIN LOOP
# -----------------------------
async def bot_loop():
    async with aiohttp.ClientSession() as session:
        while True:
            for pool in CONFIG["pools"]:
                df = await fetch_ohlcv(session, pool["network"], pool["pool_address"], CONFIG["timeframe"])
                df = compute_indicators(df)
                signal = generate_signal(df, pool["symbol"])
                if signal:
                    candle_time = df.index[-1]
                    sig_data = {
                        "time": datetime.now(timezone.utc),
                        "symbol": pool["symbol"],
                        "signal": signal,
                        "entry": df["close"].iloc[-1],
                        "sl": df["close"].iloc[-1]*0.98 if signal=="BUY" else df["close"].iloc[-1]*1.02,
                        "tp": df["close"].iloc[-1]*1.05 if signal=="BUY" else df["close"].iloc[-1]*0.95,
                        "candle_time": candle_time
                    }
                    save_signal(sig_data)
                    await send_telegram(session, f"{signal} signal for {pool['symbol']} at {sig_data['entry']}")
            await asyncio.sleep(CONFIG["poll_interval_sec"])

# -----------------------------
# START
# -----------------------------
if __name__ == "__main__":
    load_signals()
    Thread(target=run_flask, daemon=True).start()
    asyncio.run(bot_loop())
