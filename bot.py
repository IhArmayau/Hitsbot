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
        {"symbol": "ETH/USDT", "network": "eth", "pool_address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"},
        {"symbol": "BTC/USDT", "network": "eth", "pool_address": os.environ.get("BTC_POOL_ADDRESS")},
        {"symbol": "SOL/USDT", "network": "solana", "pool_address": os.environ.get("SOL_POOL_ADDRESS")}
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
# CHECK ENVIRONMENT VARIABLES
# -----------------------------
def check_env():
    logging.info("🔍 Checking environment variables...")
    for var in ["BTC_POOL_ADDRESS", "SOL_POOL_ADDRESS", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        val = os.environ.get(var)
        if val:
            logging.info(f"{var} = SET ({val[:6]}...)")
        else:
            logging.warning(f"{var} = NOT SET ❌")

# -----------------------------
# UTILITIES
# -----------------------------
def load_signals():
    if os.path.exists(CONFIG['csv_file']):
        try:
            df = pd.read_csv(CONFIG['csv_file'])
            df['time'] = pd.to_datetime(df['time'])
            df['candle_time'] = pd.to_datetime(df['candle_time'])
            global signals_memory
            signals_memory = df
            logging.info(f"📄 Loaded {len(df)} historical signals from CSV")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load CSV: {e}")

def save_signal(signal):
    global signals_memory
    signals_memory = pd.concat([signals_memory, pd.DataFrame([signal])], ignore_index=True)
    signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')
    logging.info(f"💾 Saved new signal: {signal['symbol']} {signal['signal']}")

def is_duplicate(symbol, signal, candle_time):
    candle_time = pd.to_datetime(candle_time).replace(second=0, microsecond=0)
    is_dup = not signals_memory[
        (signals_memory['symbol']==symbol) &
        (signals_memory['signal']==signal) &
        (signals_memory['candle_time']==candle_time)
    ].empty
    if is_dup:
        logging.info(f"⚠️ Duplicate detected for {symbol} {signal} at {candle_time}")
    return is_dup

async def send_telegram(session, message):
    if not CONFIG['telegram']['bot_token'] or not CONFIG['telegram']['chat_id']:
        logging.warning("❌ Telegram not configured, skipping message.")
        return
    url = f"https://api.telegram.org/bot{CONFIG['telegram']['bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG['telegram']['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logging.warning(f"⚠️ Telegram send failed: {resp.status} - {text}")
            else:
                logging.info("✅ Telegram message sent successfully.")
    except Exception as e:
        logging.error(f"❌ Telegram exception: {e}")

# -----------------------------
# FETCH OHLCV
# -----------------------------
async def fetch_ohlcv(session, network, pool_address, timeframe):
    now = time.time()
    if not pool_address:
        logging.warning(f"❌ Missing pool address for {network}")
        return pd.DataFrame()

    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
    async with semaphore:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logging.warning(f"⚠️ OHLCV fetch failed {resp.status} for {pool_address}")
                    return pd.DataFrame()
                data = await resp.json()
                df = pd.DataFrame(data.get("data", {}).get("attributes", {}).get("ohlcv_list", []))
                if df.empty:
                    logging.warning(f"⚠️ Empty OHLCV for {network}/{pool_address}/{timeframe}")
                    return pd.DataFrame()
                df.columns = ['timestamp','open','high','low','close','volume']
                for c in ['open','high','low','close','volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                logging.info(f"📊 OHLCV fetched: {pool_address} {timeframe} rows={len(df)}")
                return df.dropna()
        except Exception as e:
            logging.error(f"❌ Exception fetching OHLCV for {pool_address}: {e}")
            return pd.DataFrame()

# -----------------------------
# INDICATORS
# -----------------------------
def add_indicators(df):
    if df.empty: return pd.DataFrame()
    ind = CONFIG['indicators']
    try:
        df['ema_short'] = df['close'].ewm(span=ind['ema_short'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ind['ema_long'], adjust=False).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind['rsi_period']).rsi()
        return df.dropna()
    except Exception as e:
        logging.error(f"❌ Indicator calculation failed: {e}")
        return pd.DataFrame()

# -----------------------------
# SIGNAL LOGIC
# -----------------------------
def compute_signal(latest, prev):
    buy = sell = 0
    if prev['ema_short'] < prev['ema_long'] and latest['ema_short'] > latest['ema_long']: buy += 20
    if prev['ema_short'] > prev['ema_long'] and latest['ema_short'] < latest['ema_long']: sell += 20
    if latest['rsi'] < 40: buy += 20
    if latest['rsi'] > 60: sell += 20
    return buy, sell

# -----------------------------
# COLLECT SIGNALS
# -----------------------------
async def collect_signals(session):
    signals = []
    for pool in CONFIG['pools']:
        try:
            df = add_indicators(await fetch_ohlcv(session, pool['network'], pool['pool_address'], CONFIG['timeframes']['entry']))
            if df.empty or len(df) < 2:
                logging.info(f"ℹ️ Skipping {pool['symbol']} - insufficient data")
                continue
            latest, prev = df.iloc[-1], df.iloc[-2]
            buy, sell = compute_signal(latest, prev)
            confidence = max(buy, sell)
            if confidence < CONFIG['min_confidence']:
                logging.info(f"❌ Low confidence {confidence} for {pool['symbol']}")
                continue
            entry_signal = 'BUY' if buy > sell else 'SELL'
            candle_time = latest.name
            if is_duplicate(pool['symbol'], entry_signal, candle_time):
                continue
            entry_price = latest['close']
            sl = entry_price * (0.99 if entry_signal == 'BUY' else 1.01)
            tp = entry_price * (1.02 if entry_signal == 'BUY' else 0.98)
            signals.append({
                'time': datetime.now(timezone.utc),
                'symbol': pool['symbol'],
                'signal': entry_signal,
                'confidence': confidence,
                'strength': confidence,
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'candle_time': candle_time
            })
        except Exception as e:
            logging.error(f"❌ Error processing {pool['symbol']}: {e}")
    return signals

# -----------------------------
# MAIN LOOP
# -----------------------------
async def run_bot(session):
    load_signals()
    while True:
        logging.info("🔁 Starting new scan cycle...")
        signals = await collect_signals(session)
        logging.info(f"Cycle complete. Found {len(signals)} signal(s).")
        for s in signals:
            msg = f"*{s['symbol']}* | *{s['signal']}* | Entry: {s['entry']:.4f} | SL: {s['sl']:.4f} | TP: {s['tp']:.4f} | Confidence: {s['confidence']}%"
            logging.info(msg)
            await send_telegram(session, msg)
            save_signal(s)
        await asyncio.sleep(CONFIG['poll_interval_sec'])

# -----------------------------
# START BOT AND FLASK
# -----------------------------
async def main():
    check_env()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(run_bot(session), return_exceptions=True)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(1)
    logging.info("🚀 Starting DEX Signal Bot with Debug Logs Enabled...")
    asyncio.run(main())
