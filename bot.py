import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import logging
import os
from datetime import datetime
from flask import Flask, jsonify
from dotenv import load_dotenv
import aiohttp
import re
import aiofiles
import time
import traceback
from threading import Thread
import signal

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
    ],
    "timeframe": os.getenv("TIMEFRAME", "1h"),
    "limit": 100,
    "poll_interval_sec": 600,  # 10 minutes
    "csv_file": "signals.csv",
    "csv_save_batch": 5,
    "max_csv_signals": 1000,
    "indicators": {
        "ema_short": 9,
        "ema_long": 21,
        "ema_trend": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "atr_period": 14,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3
    },
    "smc_lookback_structure": 5,
    "smc_threshold": 0.8,
    "confidence_threshold": 50,
    "telegram": {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "rate_limit": 1
    }
}

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# GLOBALS
# -----------------------------
signals_memory = pd.DataFrame(columns=[
    'time', 'symbol', 'signal', 'entry', 'sl', 'tp', 'candle_time', 'confidence', 'ema_trend', 'rsi', 'rr'
])

# -----------------------------
# FLASK SERVER
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "KuCoin Perpetual Signal Bot is running!"

@app.route("/status")
def status():
    return jsonify(signals_memory.tail(10).to_dict(orient='records'))

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# -----------------------------
# SIGNAL BOT CLASS
# -----------------------------
class SignalBot:
    def __init__(self, config):
        self.config = config
        self.exchange = ccxt.kucoinfutures({"enableRateLimit": True})
        self.ohlcv_cache = {}
        self.semaphore = asyncio.Semaphore(5)
        self.signals_memory = signals_memory
        self.signal_buffer = []
        self.last_telegram_time = 0
        self.session = None
        self.telegram_queue = asyncio.Queue()
        self.running = True
        self.last_signal_per_symbol = {}  # For duplicate prevention

    # -----------------------------
    # TELEGRAM
    # -----------------------------
    @staticmethod
    def escape_markdown(text: str) -> str:
        return re.sub(r'([_*[\]()~`>#+-=|{}.!])', r'\\\1', text)

    async def send_telegram(self, message: str):
        await self.telegram_queue.put(message)

    async def _telegram_worker(self):
        while self.running:
            msg = await self.telegram_queue.get()
            token = self.config['telegram'].get("bot_token")
            chat_id = self.config['telegram'].get("chat_id")
            if not token or not chat_id:
                self.telegram_queue.task_done()
                continue
            now = time.time()
            if now - self.last_telegram_time < self.config['telegram']['rate_limit']:
                await asyncio.sleep(self.config['telegram']['rate_limit'] - (now - self.last_telegram_time))
            if self.session is None:
                self.session = aiohttp.ClientSession()
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": self.escape_markdown(msg), "parse_mode": "MarkdownV2"}
            for attempt in range(3):
                try:
                    async with self.session.post(url, json=payload, timeout=10) as resp:
                        if resp.status == 200:
                            self.last_telegram_time = time.time()
                            break
                except Exception as e:
                    logging.warning(f"Telegram attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
            self.telegram_queue.task_done()

    # -----------------------------
    # FETCH OHLCV & INDICATORS
    # -----------------------------
    async def fetch_ohlcv(self, symbol):
        key = (symbol, self.config["timeframe"])
        now = time.time()
        if key in self.ohlcv_cache and now - self.ohlcv_cache[key][0] < 300:
            return self.ohlcv_cache[key][1]
        for attempt in range(3):
            try:
                async with self.semaphore:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.config["timeframe"], limit=self.config["limit"])
                df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
                df["time"] = pd.to_datetime(df["time"], unit='ms', utc=True)
                df.set_index("time", inplace=True)
                df = self.add_indicators(df)
                self.ohlcv_cache[key] = (now, df)
                return df
            except Exception as e:
                logging.warning(f"Fetch OHLCV failed for {symbol}: {e}")
                await asyncio.sleep(2 ** attempt)
        return pd.DataFrame()

    def add_indicators(self, df):
        ind = self.config["indicators"]
        df['ema_short'] = df['close'].ewm(span=ind['ema_short'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ind['ema_long'], adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=ind['ema_trend'], adjust=False).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind['rsi_period']).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind['atr_period']).average_true_range()
        df['rsi_overbought_dyn'] = 70 + (df['atr'].rolling(14).mean() / df['close'].rolling(14).mean()) * 50
        df['rsi_oversold_dyn'] = 30 - (df['atr'].rolling(14).mean() / df['close'].rolling(14).mean()) * 50
        return df.dropna()

    # -----------------------------
    # SIGNAL GENERATION
    # -----------------------------
    def generate_signal(self, df, symbol):
        if df.empty or len(df)<2:
            return None
        latest = df.iloc[-1]
        ind = self.config["indicators"]
        trend = "BUY" if latest["ema_short"] > latest["ema_long"] else "SELL"
        rsi_overbought = latest.get('rsi_overbought_dyn', ind['rsi_overbought'])
        rsi_oversold = latest.get('rsi_oversold_dyn', ind['rsi_oversold'])
        if trend=="BUY" and latest["rsi"]>rsi_overbought: return None
        if trend=="SELL" and latest["rsi"]<rsi_oversold: return None

        # SMC check
        lookback = self.config.get("smc_lookback_structure",5)
        if len(df) < lookback+1: return None
        highs = df['high'].iloc[-lookback-1:]
        lows = df['low'].iloc[-lookback-1:]
        match_count = sum(
            (highs[i]>highs[i-1] and lows[i]>lows[i-1]) if trend=="BUY" else
            (lows[i]<lows[i-1] and highs[i]<highs[i-1])
            for i in range(1,len(highs))
        )
        if match_count/lookback < self.config.get("smc_threshold",0.8)*0.9: return None

        # SL/TP
        atr_ratio = latest['atr'] / df['close'].rolling(14).mean().iloc[-1]
        sl_distance = latest['atr'] * ind['atr_sl_mult'] * (1 + atr_ratio)
        tp_distance = latest['atr'] * ind['atr_tp_mult'] * (1 + atr_ratio)
        entry = latest['close']
        sl = entry - sl_distance if trend=="BUY" else entry + sl_distance
        tp = entry + tp_distance if trend=="BUY" else entry - tp_distance

        # R/R ratio
        rr = (tp - entry)/abs(entry-sl) if trend=="BUY" else (entry-tp)/abs(sl-entry)

        # Confidence
        confidence = min(max(abs(latest['ema_short']-latest['ema_long'])/latest['close']*100*10 + 
            (50 if trend=="BUY" and latest['rsi']>50 else 50 if trend=="SELL" and latest['rsi']<50 else 0),0),100)
        if confidence < self.config['confidence_threshold']: return None

        signal = {
            "time": datetime.utcnow(),
            "symbol": symbol,
            "signal": trend,
            "entry": round(entry,4),
            "sl": round(sl,4),
            "tp": round(tp,4),
            "confidence": round(confidence,2),
            "ema_trend": round(latest['ema_trend'],4),
            "rsi": round(latest['rsi'],2),
            "candle_time": df.index[-1],
            "rr": round(rr,2)
        }

        # Prevent duplicate
        last_signal = self.last_signal_per_symbol.get(symbol)
        if last_signal and last_signal['signal'] == signal['signal'] and last_signal['entry'] == signal['entry']:
            return None
        self.last_signal_per_symbol[symbol] = signal
        return signal

    # -----------------------------
    # SAVE SIGNALS
    # -----------------------------
    async def save_signal(self, signal):
        if not signal: return
        self.signal_buffer.append(signal)
        self.signals_memory = pd.concat([self.signals_memory, pd.DataFrame([signal])], ignore_index=True)
        if len(self.signals_memory) > self.config['max_csv_signals']:
            self.signals_memory = self.signals_memory.iloc[-self.config['max_csv_signals']:]
        if len(self.signal_buffer) >= self.config.get("csv_save_batch",5):
            async with aiofiles.open(self.config["csv_file"], mode='w') as f:
                await f.write(self.signals_memory.to_csv(index=False))
            self.signal_buffer.clear()

    # -----------------------------
    # MARKET STATUS
    # -----------------------------
    async def send_market_status(self):
        messages = []
        for symbol in self.config['symbols']:
            df = await self.fetch_ohlcv(symbol)
            if df.empty: continue
            latest = df.iloc[-1]
            trend = "UP" if latest['ema_short'] > latest['ema_long'] else "DOWN"
            messages.append(f"*{symbol}*\nTrend: {trend}\nPrice: {round(latest['close'],4)}\nRSI: {round(latest['rsi'],2)}")
        if messages:
            await self.send_telegram("📊 Market Status Update:\n\n" + "\n\n".join(messages))

# -----------------------------
# BOT INSTANCE
# -----------------------------
bot = SignalBot(CONFIG)

# -----------------------------
# LOAD SIGNALS
# -----------------------------
def load_signals():
    global signals_memory
    if os.path.exists(CONFIG["csv_file"]):
        try:
            df = pd.read_csv(CONFIG["csv_file"])
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
            df["candle_time"] = pd.to_datetime(df["candle_time"], errors='coerce')
            signals_memory = df
            bot.signals_memory = df
            logging.info(f"Loaded {len(df)} signals from CSV")
        except Exception as e:
            logging.warning(f"Failed to load CSV: {e}")

# -----------------------------
# MAIN BOT LOOP
# -----------------------------
async def bot_loop():
    async with aiohttp.ClientSession() as session:
        bot.session = session
        telegram_task = asyncio.create_task(bot._telegram_worker())
        while bot.running:
            new_signal_generated = False
            for symbol in CONFIG['symbols']:
                df = await bot.fetch_ohlcv(symbol)
                signal = bot.generate_signal(df, symbol)
                if signal:
                    new_signal_generated = True
                    await bot.save_signal(signal)
                    await bot.send_telegram(
                        f"*{signal['symbol']}* {signal['signal']} @ {signal['entry']}\n"
                        f"SL: {signal['sl']} | TP: {signal['tp']} | Confidence: {signal['confidence']} | R/R: {signal['rr']}"
                    )
            if not new_signal_generated:
                await bot.send_market_status()
            await asyncio.sleep(CONFIG['poll_interval_sec'])
        await bot.exchange.close()
        if bot.session:
            await bot.session.close()
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass

# -----------------------------
# GRACEFUL SHUTDOWN
# -----------------------------
def handle_exit(*args):
    bot.running = False

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# -----------------------------
# START BOT
# -----------------------------
if __name__ == "__main__":
    load_signals()
    Thread(target=run_flask, daemon=True).start()
    asyncio.run(bot_loop())
