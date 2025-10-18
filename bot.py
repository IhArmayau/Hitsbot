import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import logging
import os
from datetime import datetime, timezone
from flask import Flask
from dotenv import load_dotenv
from threading import Thread
import aiohttp

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    "symbols": [
        "ETH/USDT:USDT", 
        "INJ/USDT:USDT", 
        "ADA/USDT:USDT", 
        "XPL/USDT:USDT", 
        "XRP/USDT:USDT"
    ],  # KuCoin perpetual futures symbols
    "timeframe": os.getenv("TIMEFRAME", "1h"),  # 1m, 5m, 15m, 1h, 4h, 1d
    "limit": 100,  # Number of candles to fetch
    "poll_interval_sec": 900,  # 15 minutes
    "telegram": {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID")
    },
    "csv_file": "signals.csv"
}

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# GLOBALS
# -----------------------------
signals_memory = pd.DataFrame(columns=[
    'time', 'symbol', 'signal', 'entry', 'sl', 'tp', 'candle_time'
])

# -----------------------------
# FLASK SERVER FOR UPTIME
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "KuCoin Perpetual Signal Bot is running!"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# -----------------------------
# TELEGRAM MESSAGING
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
# FETCH OHLCV FROM KUCOIN FUTURES
# -----------------------------
async def fetch_ohlcv(exchange, symbol):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=CONFIG["timeframe"], limit=CONFIG["limit"])
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Failed to fetch {symbol}: {e}")
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
# MAIN BOT LOOP
# -----------------------------
async def bot_loop():
    exchange = ccxt.kucoinfutures({"enableRateLimit": True})
    async with aiohttp.ClientSession() as session:
        while True:
            for symbol in CONFIG["symbols"]:
                df = await fetch_ohlcv(exchange, symbol)
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
    await exchange.close()

# -----------------------------
# START BOT
# -----------------------------
if __name__ == "__main__":
    load_signals()
    Thread(target=run_flask, daemon=True).start()
    asyncio.run(bot_loop())
