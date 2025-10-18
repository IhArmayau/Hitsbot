import asyncio
import aiohttp
import pandas as pd
import ta
import logging
from datetime import datetime, timezone
from flask import Flask
from dotenv import load_dotenv
import os
from binance import AsyncClient

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    "pools": [
        {"symbol_env": os.environ.get("ETH_SYMBOL")},
        {"symbol_env": os.environ.get("INJ_SYMBOL")},
        {"symbol_env": os.environ.get("ADA_SYMBOL")},
        {"symbol_env": os.environ.get("XPL_SYMBOL")},
        {"symbol_env": os.environ.get("XRP_SYMBOL")},
    ],
    "timeframe": os.environ.get("TIMEFRAME", "1h"),  # Binance interval: 1m, 5m, 15m, 1h, 4h, 1d
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

# -----------------------------
# GLOBALS
# -----------------------------
signals_memory = pd.DataFrame(columns=[
    'time','symbol','signal','entry','sl','tp','candle_time'
])

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
# TELEGRAM INTEGRATION
# -----------------------------
async def send_telegram(session, message):
    token = CONFIG["telegram"]["bot_token"]
    chat_id = CONFIG["telegram"]["chat_id"]
    if not token or not chat_id:
        logging.warning("Telegram not configured, skipping message")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                logging.info("Telegram message sent ✅")
            else:
                text = await resp.text()
                logging.warning(f"Telegram send failed: {resp.status} {text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

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
            logging.info(f"Loaded {len(df)} signals from CSV")
        except Exception as e:
            logging.warning(f"Failed to load CSV: {e}")

def save_signal(signal):
    global signals_memory
    signals_memory = pd.concat([signals_memory, pd.DataFrame([signal])], ignore_index=True)
    signals_memory.to_csv(CONFIG["csv_file"], index=False)
    logging.info(f"Saved signal: {signal['symbol']} {signal['signal']}")

# -----------------------------
# FETCH OHLCV FROM BINANCE
# -----------------------------
async def fetch_ohlcv(client, symbol, interval):
    """
    Fetch OHLCV data from Binance.
    Returns a DataFrame with timestamp, open, high, low, close, volume.
    """
    try:
        klines = await client.get_klines(symbol=symbol, interval=interval, limit=500)
        df = pd.DataFrame(klines, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col])
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        logging.error(f"Binance fetch error for {symbol}: {e}")
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
    client = await AsyncClient.create()
    async with aiohttp.ClientSession() as session:
        while True:
            for pool in CONFIG["pools"]:
                symbol = pool["symbol_env"]
                df = await fetch_ohlcv(client, symbol, CONFIG["timeframe"])
                df = compute_indicators(df)
                signal = generate_signal(df, symbol)
                if signal:
                    candle_time = df.index[-1]
                    sig_data = {
                        "time": datetime.now(timezone.utc),
                        "symbol": symbol,
                        "signal": signal,
                        "entry": df["close"].iloc[-1],
                        "sl": df["close"].iloc[-1]*0.98 if signal=="BUY" else df["close"].iloc[-1]*1.02,
                        "tp": df["close"].iloc[-1]*1.05 if signal=="BUY" else df["close"].iloc[-1]*0.95,
                        "candle_time": candle_time
                    }
                    save_signal(sig_data)
                    await send_telegram(session, f"{signal} signal for {symbol} at {sig_data['entry']}")
            await asyncio.sleep(CONFIG["poll_interval_sec"])

# -----------------------------
# START
# -----------------------------
if __name__ == "__main__":
    load_signals()
    # Start Flask server in background
    from threading import Thread
    Thread(target=run_flask, daemon=True).start()
    # Start async bot
    asyncio.run(bot_loop())
