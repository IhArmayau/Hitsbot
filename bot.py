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
        # ETH pool (example known address)
        {"symbol": "ETH/USDT", "network": "eth", "pool_address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"},
        # BTC and SOL addresses pulled from env (may be None if not set)
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
# ENV CHECK
# -----------------------------
def check_env():
    logging.info("🔍 Checking environment variables...")
    for var in ["BTC_POOL_ADDRESS", "SOL_POOL_ADDRESS", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        val = os.environ.get(var)
        if val:
            # don't print full tokens/IDs — show short prefix for verification
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
        # Normalize expected columns: accept either 'time' or maybe 'timestamp', and ensure 'candle_time' present
        if 'time' not in df.columns and 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'time'})
        if 'candle_time' not in df.columns and 'candle_timestamp' in df.columns:
            df = df.rename(columns={'candle_timestamp': 'candle_time'})

        # If still missing columns, try to infer or create them safely
        if 'time' not in df.columns:
            logging.warning("Loaded CSV missing 'time' column — creating 'time' from current time for rows.")
            df['time'] = pd.Timestamp.now(tz=timezone.utc)
        if 'candle_time' not in df.columns:
            logging.warning("Loaded CSV missing 'candle_time' column — attempting to create from 'time'.")
            df['candle_time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

        # Parse datetimes robustly
        df['time'] = pd.to_datetime(df['time'], errors='coerce').fillna(pd.Timestamp.now(tz=timezone.utc))
        df['candle_time'] = pd.to_datetime(df['candle_time'], errors='coerce').fillna(df['time'])

        # Ensure other expected columns exist (fill defaults if required)
        for col in ['symbol','signal','confidence','strength','entry','sl','tp']:
            if col not in df.columns:
                logging.warning(f"Loaded CSV missing '{col}' column — filling with defaults.")
                df[col] = None

        signals_memory = df[[
            'time','symbol','signal','confidence','strength','entry','sl','tp','candle_time'
        ]].copy()
        logging.info(f"📄 Loaded {len(signals_memory)} historical signals from CSV")
    except Exception as e:
        logging.warning(f"⚠️ Failed to load CSV (safe fallback) — starting with empty memory: {e}")
        signals_memory = pd.DataFrame(columns=[
            'time','symbol','signal','confidence','strength','entry','sl','tp','candle_time'
        ])

def save_signal(signal):
    global signals_memory
    try:
        # Ensure candle_time and time are serializable
        s = signal.copy()
        if isinstance(s.get('time'), (pd.Timestamp, datetime)):
            s['time'] = pd.to_datetime(s['time']).isoformat()
        if isinstance(s.get('candle_time'), (pd.Timestamp, datetime)):
            s['candle_time'] = pd.to_datetime(s['candle_time']).isoformat()

        signals_memory = pd.concat([signals_memory, pd.DataFrame([s])], ignore_index=True)
        # keep only last max_rows
        if len(signals_memory) > CONFIG['max_rows']:
            signals_memory = signals_memory.iloc[-CONFIG['max_rows']:]
        signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')
        logging.info(f"💾 Saved signal to CSV: {s.get('symbol')} {s.get('signal')}")
    except Exception as e:
        logging.error(f"❌ Failed to save signal: {e}")

def prune_signals():
    global signals_memory
    if len(signals_memory) > CONFIG['max_rows']:
        signals_memory = signals_memory.iloc[-CONFIG['max_rows']:]
        signals_memory.to_csv(CONFIG['csv_file'], index=False, float_format='%.6f')

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
# TELEGRAM SENDING (only used for actual signals)
# -----------------------------
async def send_telegram(session, message):
    if not CONFIG['telegram']['bot_token'] or not CONFIG['telegram']['chat_id']:
        logging.warning("❌ Telegram not configured, skipping message.")
        return
    url = f"https://api.telegram.org/bot{CONFIG['telegram']['bot_token']}/sendMessage"
    payload = {"chat_id": CONFIG['telegram']['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            text = await resp.text()
            if resp.status != 200:
                logging.warning(f"⚠️ Telegram send failed: {resp.status} - {text}")
            else:
                logging.info("✅ Telegram message sent successfully.")
    except Exception as e:
        logging.error(f"❌ Telegram exception: {e}")

# -----------------------------
# FETCH OHLCV (GeckoTerminal)
# -----------------------------
async def fetch_ohlcv(session, network, pool_address, timeframe):
    now = time.time()
    if not pool_address:
        logging.warning(f"❌ Missing pool address for network '{network}' — skipping fetch.")
        return pd.DataFrame()

    key = (network, pool_address, timeframe)
    # Caching
    if key in ohlcv_cache and now - ohlcv_cache[key][0] < CONFIG['cache_ttl_sec']:
        logging.info(f"Using cached OHLCV for {pool_address} {timeframe} (age {now - ohlcv_cache[key][0]:.1f}s)")
        return ohlcv_cache[key][1].copy()

    async with semaphore:
        url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
        try:
            logging.info(f"Fetching OHLCV: {network}/{pool_address}/{timeframe}")
            async with session.get(url, timeout=15) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logging.warning(f"⚠️ GeckoTerminal returned status {resp.status} for {pool_address} (text: {text[:200]})")
                    pool_failures[pool_address] = pool_failures.get(pool_address, 0) + 1
                    if pool_failures[pool_address] >= CONFIG['max_failures']:
                        pool_skip_until[pool_address] = now + CONFIG['skip_duration']
                        logging.warning(f"⏸ Skipping pool {pool_address} until {pool_skip_until[pool_address]} due to repeated failures")
                    return pd.DataFrame()

                # Try to parse JSON
                try:
                    data = await session._loop.run_in_executor(None, lambda: pd.io.json.loads(text))
                except Exception:
                    # Fallback: use session.json() which will reparse; but keep robust logging
                    try:
                        data = await resp.json()
                    except Exception as e:
                        logging.error(f"❌ Failed to parse JSON from GeckoTerminal for {pool_address}: {e}")
                        logging.debug(f"Raw text (truncated): {text[:500]}")
                        return pd.DataFrame()

                # Extract ohlcv_list safely
                ohlcv_list = data.get("data", {}).get("attributes", {}).get("ohlcv_list", None)
                if ohlcv_list is None:
                    logging.warning(f"⚠️ GeckoTerminal response missing 'ohlcv_list' for {pool_address}. Raw: {text[:400]}")
                    return pd.DataFrame()
                if not isinstance(ohlcv_list, list) or len(ohlcv_list) == 0:
                    logging.warning(f"⚠️ Empty or non-list 'ohlcv_list' for {pool_address}/{timeframe}.")
                    return pd.DataFrame()

                # Build DataFrame from list of lists or list of dicts
                df = pd.DataFrame(ohlcv_list)
                # If list of values (no keys), expect 6 columns: [timestamp, open, high, low, close, volume]
                if df.shape[1] == 6 and not set(df.columns).intersection({'timestamp','time','open','close'}):
                    df.columns = ['timestamp','open','high','low','close','volume']
                # If object/dict columns exist, try to select known names
                if 'timestamp' not in df.columns and 'time' in df.columns:
                    df = df.rename(columns={'time': 'timestamp'})

                # Convert numeric columns
                for c in ['open','high','low','close','volume']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                # Convert timestamp to datetime index
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    except Exception:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    df.set_index('timestamp', inplace=True)
                else:
                    logging.warning(f"⚠️ No timestamp column found after parsing for {pool_address}. Columns: {df.columns.tolist()}")
                    return pd.DataFrame()

                df = df.sort_index()
                if df.empty:
                    logging.warning(f"⚠️ Parsed OHLCV DataFrame empty for {pool_address}")
                    return pd.DataFrame()

                ohlcv_cache[key] = (now, df.copy())
                pool_failures[pool_address] = 0
                logging.info(f"📊 OHLCV fetched: {pool_address} {timeframe} rows={len(df)}")
                return df.dropna()
        except asyncio.TimeoutError:
            logging.error(f"❌ Timeout fetching OHLCV for {pool_address}")
            pool_failures[pool_address] = pool_failures.get(pool_address, 0) + 1
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"❌ Exception fetching OHLCV for {pool_address}: {e}")
            pool_failures[pool_address] = pool_failures.get(pool_address, 0) + 1
            return pd.DataFrame()

# -----------------------------
# INDICATORS (minimal & robust)
# -----------------------------
def add_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    ind = CONFIG['indicators']
    try:
        df = df.copy()
        df['ema_short'] = df['close'].ewm(span=ind['ema_short'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ind['ema_long'], adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=ind['ema_trend'], adjust=False).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind['rsi_period']).rsi()
        # Keep ADX/MACD/BB/ATR only if needed; compute lightly to avoid heavy exceptions
        try:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind['adx_period'])
            df['adx'] = adx.adx()
            df['plus_di'] = adx.adx_pos()
            df['minus_di'] = adx.adx_neg()
        except Exception:
            logging.debug("Optional ADX failed (continuing)")

        try:
            macd = ta.trend.MACD(df['close'], ind['macd_fast'], ind['macd_slow'], ind['macd_signal'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
        except Exception:
            logging.debug("Optional MACD failed (continuing)")

        try:
            bb = ta.volatility.BollingerBands(df['close'], ind['bb_window'], ind['bb_std'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
        except Exception:
            logging.debug("Optional BollingerBands failed (continuing)")

        try:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind['atr_period']).average_true_range()
        except Exception:
            logging.debug("Optional ATR failed (continuing)")

        return df.dropna()
    except Exception as e:
        logging.error(f"❌ Indicator calc failed: {e}")
        return pd.DataFrame()

# -----------------------------
# SIGNAL LOGIC (kept simple & deterministic)
# -----------------------------
def compute_signal(latest, prev):
    buy = sell = 0
    try:
        if prev['ema_short'] < prev['ema_long'] and latest['ema_short'] > latest['ema_long']:
            buy += 20
        if prev['ema_short'] > prev['ema_long'] and latest['ema_short'] < latest['ema_long']:
            sell += 20
        if 'macd' in latest.index and 'macd_signal' in latest.index:
            if prev.get('macd', 0) < prev.get('macd_signal', 0) and latest.get('macd', 0) > latest.get('macd_signal', 0):
                buy += 20
            if prev.get('macd', 0) > prev.get('macd_signal', 0) and latest.get('macd', 0) < latest.get('macd_signal', 0):
                sell += 20
        if latest.get('rsi', 0) < CONFIG['indicators']['rsi_oversold']:
            buy += 20
        if latest.get('rsi', 0) > CONFIG['indicators']['rsi_overbought']:
            sell += 20
        if latest.get('adx', 0) > CONFIG['indicators']['adx_threshold']:
            if latest.get('plus_di', 0) > latest.get('minus_di', 0):
                buy += 15
            if latest.get('minus_di', 0) > latest.get('plus_di', 0):
                sell += 15
        if latest.get('bb_lower') is not None and latest.get('close') < latest.get('bb_lower'):
            buy += 15
        if latest.get('bb_upper') is not None and latest.get('close') > latest.get('bb_upper'):
            sell += 15
    except Exception as e:
        logging.error(f"Error in compute_signal: {e}")
    return buy, sell

# -----------------------------
# COLLECT SIGNALS
# -----------------------------
async def collect_signals(session):
    signals = []
    for pool in CONFIG['pools']:
        try:
            # Skip pools with no address configured
            if not pool.get('pool_address'):
                logging.info(f"Skipping {pool['symbol']} — pool_address not configured.")
                continue

            df_entry = add_indicators(await fetch_ohlcv(session, pool['network'], pool['pool_address'], CONFIG['timeframes']['entry']))
            if df_entry.empty or len(df_entry) < 2:
                logging.info(f"ℹ️ Skipping {pool['symbol']} - insufficient entry timeframe data")
                continue

            latest, prev = df_entry.iloc[-1], df_entry.iloc[-2]
            buy, sell = compute_signal(latest, prev)
            confidence = max(buy, sell)
            if confidence < CONFIG['min_confidence']:
                logging.info(f"❌ Low confidence {confidence} for {pool['symbol']}")
                continue

            entry_signal = 'BUY' if buy > sell else 'SELL'
            candle_time = latest.name.replace(second=0, microsecond=0) if hasattr(latest.name, 'replace') else latest.name

            if is_duplicate(pool['symbol'], entry_signal, candle_time):
                continue

            # Confirmations: check 1h and 4h agree if data available
            confirmed = True
            for tf_key in ['confirmation_1h', 'confirmation_4h']:
                tf = CONFIG['timeframes'][tf_key]
                df_tf = add_indicators(await fetch_ohlcv(session, pool['network'], pool['pool_address'], tf))
                if df_tf.empty or len(df_tf) < 2:
                    logging.info(f"ℹ️ No confirmation timeframe data ({tf}) for {pool['symbol']} — skipping confirmation for this tf")
                    continue
                l, p = df_tf.iloc[-1], df_tf.iloc[-2]
                buy_tf, sell_tf = compute_signal(l, p)
                signal_tf = 'BUY' if buy_tf > sell_tf else 'SELL'
                if signal_tf != entry_signal:
                    confirmed = False
                    logging.info(f"❌ Confirmation mismatch on {tf} for {pool['symbol']} (entry {entry_signal} vs {signal_tf})")
                    break

            if not confirmed:
                continue

            entry_price = latest['close']
            atr = latest.get('atr', None)
            if atr is None or pd.isna(atr):
                # fallback tiny SL/TP if ATR not present
                sl = entry_price * (0.995 if entry_signal == 'BUY' else 1.005)
                tp = entry_price * (1.01 if entry_signal == 'BUY' else 0.99)
            else:
                sl = entry_price - atr * CONFIG['indicators']['atr_sl'] if entry_signal == 'BUY' else entry_price + atr * CONFIG['indicators']['atr_sl']
                tp = entry_price + atr * CONFIG['indicators']['atr_tp'] if entry_signal == 'BUY' else entry_price - atr * CONFIG['indicators']['atr_tp']

            signals.append({
                'time': datetime.now(timezone.utc),
                'symbol': pool['symbol'],
                'signal': entry_signal,
                'confidence': confidence,
                'strength': confidence,
                'entry': float(entry_price),
                'sl': float(sl),
                'tp': float(tp),
                'candle_time': candle_time
            })
            logging.info(f"⭐ Candidate signal: {pool['symbol']} {entry_signal} conf={confidence} entry={entry_price:.6f}")

        except Exception as e:
            logging.error(f"❌ Error processing {pool.get('symbol','unknown')}: {e}")

    # sort and return top signals
    signals_sorted = sorted(signals, key=lambda x: x['confidence'], reverse=True)[:CONFIG['top_signals']]
    return signals_sorted

# -----------------------------
# MARKET STATUS TASK (sends periodic short messages IF Telegram configured)
# -----------------------------
async def send_market_status(session):
    while True:
        try:
            for pool in CONFIG['pools']:
                if not pool.get('pool_address'):
                    continue
                df = add_indicators(await fetch_ohlcv(session, pool['network'], pool['pool_address'], CONFIG['timeframes']['entry']))
                if df.empty:
                    continue
                latest = df.iloc[-1]
                msg = f"*{pool['symbol']}* | Price: {latest['close']:.6f} | EMA Short: {latest['ema_short']:.6f} | EMA Long: {latest['ema_long']:.6f} | RSI: {latest['rsi']:.2f}"
                # send only if Telegram configured (this is not debug spam)
                await send_telegram(session, msg)
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
            # resume any skipped pools if their skip time has elapsed
            for pool_addr in list(pool_skip_until.keys()):
                if time.time() >= pool_skip_until[pool_addr]:
                    logging.info(f"Resuming pool {pool_addr}")
                    del pool_skip_until[pool_addr]
                    pool_failures[pool_addr] = 0

            logging.info("🔁 Starting new scan cycle...")
            signals = await collect_signals(session)
            logging.info(f"Cycle complete. Found {len(signals)} top signal(s).")
            for s in signals:
                msg = f"*{s['symbol']}* | *{s['signal']}* | Entry: {s['entry']:.6f} | SL: {s['sl']:.6f} | TP: {s['tp']:.6f} | Confidence: {s['confidence']}%"
                logging.info(f"Dispatching signal: {msg}")
                await send_telegram(session, msg)
                save_signal(s)
            prune_signals()
        except Exception as e:
            logging.error(f"Critical exception in main loop: {e}")
        await asyncio.sleep(CONFIG['poll_interval_sec'])

# -----------------------------
# START BOT AND FLASK
# -----------------------------
async def main():
    check_env()
    async with aiohttp.ClientSession() as session:
        # Run both bot and market status in parallel; allow exceptions to surface to logs
        await asyncio.gather(
            run_bot(session),
            send_market_status(session),
            return_exceptions=True
        )

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(1)
    logging.info("🚀 Starting DEX Signal Bot (GeckoTerminal) with robust debug logs...")
    asyncio.run(main())
