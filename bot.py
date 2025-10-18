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
    logging.info("Starting Flask web server on port 10000...")
    app.run(host="0.0.0.0", port=10000)

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
            logging.info(f"Loaded {len(df)} historical signals")
        except Exception as e:
            logging.warning(f"Failed to load CSV: {e}")

def save_signal(signal):
    global signals_memory
    signals_memory = pd.concat([signals_memory, pd.DataFrame([signal])], ignore_index=True)
    signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')

def prune_signals():
    global signals_memory
    if len(signals_memory) > CONFIG['max_rows']:
        signals_memory = signals_memory.iloc[-CONFIG['max_rows']:]
        signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')

def is_duplicate(symbol, signal, candle_time):
    candle_time = pd.to_datetime(candle_time).replace(second=0, microsecond=0)
    return not signals_memory[
        (signals_memory['symbol']==symbol) &
        (signals_memory['signal']==signal) &
        (signals_memory['candle_time']==candle_time)
    ].empty

async def send_telegram(session, message):
    if not CONFIG['telegram']['bot_token'] or not CONFIG['telegram']['chat_id']:
        logging.warning("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{CONFIG['telegram']['bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG['telegram']['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                logging.warning(f"Telegram failed: {resp.status} - {await resp.text()}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

# -----------------------------
# FETCH OHLCV
# -----------------------------
async def fetch_ohlcv(session, network, pool_address, timeframe):
    now = time.time()
    if pool_address in pool_skip_until and now < pool_skip_until[pool_address]:
        return ohlcv_cache.get((network,pool_address,timeframe),(0,pd.DataFrame()))[1].copy()

    key = (network,pool_address,timeframe)
    if key in ohlcv_cache and now - ohlcv_cache[key][0] < CONFIG['cache_ttl_sec']:
        return ohlcv_cache[key][1].copy()

    async with semaphore:
        url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    pool_failures[pool_address] = pool_failures.get(pool_address,0)+1
                    if pool_failures[pool_address]>=CONFIG['max_failures']:
                        pool_skip_until[pool_address] = now + CONFIG['skip_duration']
                    return ohlcv_cache.get(key,(0,pd.DataFrame()))[1].copy()
                data = await resp.json()
                df = pd.DataFrame(data.get("data", {}).get("attributes", {}).get("ohlcv_list", []))
                if df.empty: return pd.DataFrame()
                df.columns = ['timestamp','open','high','low','close','volume']
                for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
                df['timestamp']=pd.to_datetime(df['timestamp'],unit='s')
                df.set_index('timestamp', inplace=True)
                ohlcv_cache[key] = (now, df.copy())
                pool_failures[pool_address]=0
                return df.dropna()
        except Exception as e:
            pool_failures[pool_address] = pool_failures.get(pool_address,0)+1
            if pool_failures[pool_address]>=CONFIG['max_failures']:
                pool_skip_until[pool_address]=now+CONFIG['skip_duration']
            logging.error(f"fetch_ohlcv exception for {pool_address}: {e}")
            return ohlcv_cache.get(key,(0,pd.DataFrame()))[1].copy()

# -----------------------------
# INDICATORS
# -----------------------------
def add_indicators(df):
    if df.empty: return pd.DataFrame()
    ind = CONFIG['indicators']
    df['ema_short'] = df['close'].ewm(span=ind['ema_short'], adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind['ema_long'], adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=ind['ema_trend'], adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind['rsi_period']).rsi()
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind['adx_period'])
    df['adx'] = adx.adx()
    df['plus_di'] = adx.adx_pos()
    df['minus_di'] = adx.adx_neg()
    macd = ta.trend.MACD(df['close'], ind['macd_fast'], ind['macd_slow'], ind['macd_signal'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'], ind['bb_window'], ind['bb_std'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind['atr_period']).average_true_range()
    return df.dropna()

# -----------------------------
# SIGNAL LOGIC
# -----------------------------
def compute_signal(latest, prev):
    buy = sell = 0
    if prev['ema_short']<prev['ema_long'] and latest['ema_short']>latest['ema_long']: buy+=20
    if prev['ema_short']>prev['ema_long'] and latest['ema_short']<latest['ema_long']: sell+=20
    if prev['macd']<prev['macd_signal'] and latest['macd']>latest['macd_signal']: buy+=20
    if prev['macd']>prev['macd_signal'] and latest['macd']<latest['macd_signal']: sell+=20
    if latest['rsi']<CONFIG['indicators']['rsi_oversold']: buy+=20
    if latest['rsi']>CONFIG['indicators']['rsi_overbought']: sell+=20
    if latest['adx']>CONFIG['indicators']['adx_threshold']:
        if latest['plus_di']>latest['minus_di']: buy+=15
        if latest['minus_di']>latest['plus_di']: sell+=15
    if latest['close']<latest['bb_lower']: buy+=15
    if latest['close']>latest['bb_upper']: sell+=15
    return buy,sell

# -----------------------------
# COLLECT SIGNALS
# -----------------------------
async def collect_signals(session):
    signals=[]
    for pool in CONFIG['pools']:
        try:
            df_entry=add_indicators(await fetch_ohlcv(session,pool['network'],pool['pool_address'],CONFIG['timeframes']['entry']))
            if df_entry.empty or len(df_entry)<2: continue
            latest, prev = df_entry.iloc[-1], df_entry.iloc[-2]
            buy, sell = compute_signal(latest, prev)
            confidence = max(buy, sell)
            if confidence < CONFIG['min_confidence']: continue
            entry_signal = 'BUY' if buy>sell else 'SELL'
            candle_time = latest.name.replace(second=0, microsecond=0)
            if is_duplicate(pool['symbol'], entry_signal, candle_time): continue

            # Confirmations 1h & 4h
            confirmed=True
            for tf in ['confirmation_1h','confirmation_4h']:
                df_tf=add_indicators(await fetch_ohlcv(session,pool['network'],pool['pool_address'],CONFIG['timeframes'][tf]))
                if df_tf.empty or len(df_tf)<2: continue
                l, p=df_tf.iloc[-1], df_tf.iloc[-2]
                buy_tf,sell_tf=compute_signal(l,p)
                signal_tf='BUY' if buy_tf>sell_tf else 'SELL'
                if signal_tf!=entry_signal: confirmed=False
            if not confirmed: continue

            # Levels
            entry_price=latest['close']
            atr=latest['atr']
            sl = entry_price-atr*CONFIG['indicators']['atr_sl'] if entry_signal=='BUY' else entry_price+atr*CONFIG['indicators']['atr_sl']
            tp = entry_price+atr*CONFIG['indicators']['atr_tp'] if entry_signal=='BUY' else entry_price-atr*CONFIG['indicators']['atr_tp']

            signals.append({
                'time':datetime.now(timezone.utc),
                'symbol':pool['symbol'],
                'signal':entry_signal,
                'confidence':confidence,
                'strength':confidence,
                'entry':entry_price,
                'sl':sl,
                'tp':tp,
                'candle_time':candle_time
            })
        except Exception as e:
            logging.error(f"Error processing {pool['symbol']}: {e}")
    return sorted(signals,key=lambda x:x['confidence'],reverse=True)[:CONFIG['top_signals']]

# -----------------------------
# MARKET STATUS TASK
# -----------------------------
async def send_market_status(session):
    while True:
        try:
            for pool in CONFIG['pools']:
                df=add_indicators(await fetch_ohlcv(session,pool['network'],pool['pool_address'],CONFIG['timeframes']['entry']))
                if df.empty: continue
                latest=df.iloc[-1]
                msg=f"*{pool['symbol']}* | Price: {latest['close']:.6f} | EMA Short: {latest['ema_short']:.6f} | EMA Long: {latest['ema_long']:.6f} | RSI: {latest['rsi']:.2f}"
                await send_telegram(session,msg)
        except Exception as e:
            logging.error(f"Market status update failed: {e}")
        await asyncio.sleep(CONFIG['status_interval_sec'])

# -----------------------------
# MAIN BOT LOOP
# -----------------------------
async def run_bot(session):
    load_signals()
    while True:
        try:
            for pool_addr in list(pool_skip_until):
                if time.time()>=pool_skip_until[pool_addr]:
                    logging.info(f"Resuming pool {pool_addr}")
                    del pool_skip_until[pool_addr]
                    pool_failures[pool_addr]=0
            signals=await collect_signals(session)
            logging.info(f"Cycle complete. {len(signals)} top signals found.")
            for s in signals:
                msg=f"*{s['symbol']}* | *{s['signal']}* | Entry: {s['entry']:.6f} | SL: {s['sl']:.6f} | TP: {s['tp']:.6f} | Confidence: {s['confidence']}%"
                logging.info(msg)
                await send_telegram(session,msg)
                save_signal(s)
            prune_signals()
        except Exception as e:
            logging.error(f"Critical exception: {e}")
        await asyncio.sleep(CONFIG['poll_interval_sec'])

# -----------------------------
# START BOT AND FLASK
# -----------------------------
async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            run_bot(session),
            send_market_status(session)
        )

if __name__=="__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(1)
    logging.info("Starting DEX Perpetual Signal Bot (15m/1h/4h)...")
    asyncio.run(main())
