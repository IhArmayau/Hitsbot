#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import aiosqlite
import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone
import numpy as np
import aiohttp
import pickle
from dotenv import load_dotenv
import signal
import sys

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Logging Configuration
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SignalBotAI")

# -----------------------------
# Configuration Classes
# -----------------------------
@dataclass
class IndicatorsConfig:
    ema_short: int = int(os.getenv("EMA_SHORT", 9))
    ema_medium: int = int(os.getenv("EMA_MEDIUM", 21))
    ema_long: int = int(os.getenv("EMA_LONG", 50))
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    rsi_overbought: float = float(os.getenv("RSI_OVERBOUGHT", 70))
    rsi_oversold: float = float(os.getenv("RSI_OVERSOLD", 30))
    atr_period: int = int(os.getenv("ATR_PERIOD", 14))
    atr_tp_mult: float = float(os.getenv("ATR_TP_MULT", 3.0))
    atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", 1.5))
    adx_period: int = int(os.getenv("ADX_PERIOD", 14))
    adx_threshold: float = float(os.getenv("ADX_THRESHOLD", 20))
    bb_period: int = int(os.getenv("BB_PERIOD", 20))
    bb_std: float = float(os.getenv("BB_STD", 2.0))

@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: os.getenv(
        "SYMBOLS",
        "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,XPL/USDT,INJ/USDT"
    ).split(","))
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "4h")
    limit: int = int(os.getenv("LIMIT", 200))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 70))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "xgb_model.pkl")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    uptimerobot_url: Optional[str] = os.getenv("UPTIMEROBOT_URL")
    heartbeat_interval: int = int(os.getenv("HEARTBEAT_INTERVAL", 60))

# -----------------------------
# Database Management
# -----------------------------
class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def init_db(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                confidence REAL,
                rr REAL,
                outcome TEXT
            )
        """)
        await self.conn.commit()

    async def insert_signal(self, sig: Dict):
        if not self.conn:
            logger.warning("Database connection not initialized.")
            return
        try:
            await self.conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal, entry, sl, tp, confidence, rr, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig['timestamp'], sig['symbol'], sig['signal'], sig['entry'],
                sig['sl'], sig['tp'], sig['confidence'], sig['rr'], sig.get('outcome')
            ))
            await self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert signal into DB: {e}")

    async def close(self):
        if self.conn:
            await self.conn.close()

# -----------------------------
# Performance Metrics
# -----------------------------
class PerformanceMetrics:
    def __init__(self):
        self.metrics: Dict[str, Dict] = {}

    def update(self, sig: Dict):
        sym = sig['symbol']
        if sym not in self.metrics:
            self.metrics[sym] = {"total":0, "wins":0, "losses":0, "avg_rr":0.0, "avg_conf":0.0, "win_rate":0.0}
        m = self.metrics[sym]
        m['total'] += 1
        if sig.get("outcome") == "WIN": m['wins'] += 1
        elif sig.get("outcome") == "LOSS": m['losses'] += 1
        m['avg_rr'] = ((m['avg_rr']*(m['total']-1)) + sig['rr']) / m['total']
        m['avg_conf'] = ((m['avg_conf']*(m['total']-1)) + sig['confidence']) / m['total']
        m['win_rate'] = (m['wins'] / m['total'] * 100) if m['total'] > 0 else 0

# -----------------------------
# Telegram Utilities
# -----------------------------
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data={"chat_id": chat_id, "text": message, "parse_mode":"Markdown"}) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram send failed with status: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

# -----------------------------
# Heartbeat
# -----------------------------
async def send_heartbeat(url: str):
    if not url:
        return
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"Heartbeat failed with status: {resp.status}")
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

async def heartbeat_loop(url: str, interval: int):
    while True:
        await send_heartbeat(url)
        await asyncio.sleep(interval)

# -----------------------------
# Candle Confirmation
# -----------------------------
def confirm_candle(df: pd.DataFrame, sig_type: str) -> bool:
    if len(df) < 3: 
        return False
    prev2, prev1, curr = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if sig_type == "BUY" and curr['close'] > prev1['close'] > prev2['close'] and curr['open'] < prev1['close']:
        return True
    if sig_type == "SELL" and curr['close'] < prev1['close'] < prev2['close'] and curr['open'] > prev1['close']:
        return True
    return False

# -----------------------------
# Visualization Helpers
# -----------------------------
def rr_bar(rr_value: float, max_rr: float = 5.0, length: int = 10) -> str:
    blocks = min(int((rr_value / max_rr) * length), length)
    empty_blocks = length - blocks
    return "🟩" * blocks + "⬜" * empty_blocks

def format_signal_message(signals: List[Dict]) -> str:
    if not signals:
        return "_No new signals at this time._"
    msg_lines = ["*📊 SignalBotAI New Signals* 🤖\n"]
    for sig in signals:
        signal_type = sig["signal"]
        symbol = sig["symbol"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        rr = sig["rr"]
        confidence = sig["confidence"]

        if confidence >= 90:
            signal_icon = "🟢" if signal_type == "BUY" else "🔴"
        elif confidence >= 70:
            signal_icon = "🟡"
        else:
            signal_icon = "⚪"

        htf_arrow = "📈" if sig.get("htf_trend", 0) > 0 else "📉"
        bb_emoji = "🔵" if sig.get("bb_trend", 0) > 0 else "🔴"
        rr_visual = rr_bar(rr)

        msg_lines.append(
            f"{bb_emoji} {signal_icon} *{signal_type}* {htf_arrow}\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Entry:* `{entry:.2f}`\n"
            f"• *SL:* `{sl:.2f}` | *TP:* `{tp:.2f}`\n"
            f"• *R:R:* `{rr:.2f}` {rr_visual}\n"
            f"• *Confidence:* `{confidence:.2f}%`\n"
            f"────────────────────────"
        )
    return "\n".join(msg_lines)

# -----------------------------
# SignalBot Class
# -----------------------------
class SignalBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.exchange = ccxt.kucoinfutures({"enableRateLimit": True})
        self.store = SignalStore(cfg.sqlite_db)
        self.metrics = PerformanceMetrics()
        self.running_event = asyncio.Event()
        self.running_event.set()
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.ai_model = self.load_ai_model()

    def load_ai_model(self):
        if os.path.exists(self.cfg.ml_model_path):
            try:
                with open(self.cfg.ml_model_path, "rb") as f:
                    model = pickle.load(f)
                    logger.info("AI model loaded successfully.")
                    return model
            except Exception as e:
                logger.error(f"Failed to load AI model: {e}")
        logger.warning("No AI model found. Running without AI predictions.")
        return None

    async def fetch_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=self.cfg.limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                return df
            except Exception as e:
                logger.warning(f"Fetch candles failed ({symbol}, {tf}) attempt {attempt+1}: {e}")
                await asyncio.sleep(1.5**attempt + random.random())
        return pd.DataFrame()

    def add_indicators(self, df: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        ind = self.cfg.indicators
        df['ema_short'] = df['close'].ewm(span=ind.ema_short, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=ind.ema_medium, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ind.ema_long, adjust=False).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind.rsi_period).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind.atr_period).average_true_range()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind.adx_period).adx()
        bb = ta.volatility.BollingerBands(df['close'], ind.bb_period, ind.bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)
        df['vol_avg'] = df['volume'].rolling(20).mean()
        df['vol_ok'] = df['volume'] > df['vol_avg']
        if df_htf is not None and not df_htf.empty:
            df['htf_trend'] = np.where(df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1], 1, -1)
        else:
            df['htf_trend'] = 0
        df.dropna(inplace=True)
        return df

    def ai_predict(self, row: pd.Series) -> float:
        if self.ai_model is None:
            return 0.5
        try:
            features_list = ['ema_short','ema_medium','ema_long','rsi','atr','adx','bb_trend','vol_ok','htf_trend']
            features = [row[f] for f in features_list if f in row]
            features = np.array(features).reshape(1, -1)
            prob = float(self.ai_model.predict_proba(features)[0][1])
            logger.debug(f"AI prediction: {prob:.2f}")
            return prob
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            return 0.5

    async def generate_signal(self, symbol: str):
        async with self.semaphore:
            df = await self.fetch_candles(symbol, self.cfg.timeframe)
            df_htf = await self.fetch_candles(symbol, self.cfg.higher_timeframe)
            if df.empty or df_htf.empty:
                return
            df = self.add_indicators(df, df_htf)
            last_row = df.iloc[-1]

            signal_type = None
            if last_row['close'] > last_row['ema_short'] > last_row['ema_medium']:
                if confirm_candle(df, "BUY"): signal_type = "BUY"
            elif last_row['close'] < last_row['ema_short'] < last_row['ema_medium']:
                if confirm_candle(df, "SELL"): signal_type = "SELL"
            if not signal_type: return

            confidence = self.ai_predict(last_row) * 100
            if confidence < self.cfg.confidence_threshold: return

            atr = last_row['atr']
            entry = last_row['close']
            if signal_type == "BUY":
                sl = entry - atr * self.cfg.indicators.atr_sl_mult
                tp = entry + atr * self.cfg.indicators.atr_tp_mult
            else:
                sl = entry + atr * self.cfg.indicators.atr_sl_mult
                tp = entry - atr * self.cfg.indicators.atr_tp_mult
            rr = abs((tp - entry) / (entry - sl)) if (entry - sl) != 0 else 0

            signal_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal": signal_type,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
                "rr": rr,
                "htf_trend": last_row['htf_trend'],
                "bb_trend": last_row['bb_trend']
            }

            await self.store.insert_signal(signal_data)
            self.metrics.update(signal_data)
            logger.info(f"Signal generated: {signal_data}")

            if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                msg = format_signal_message([signal_data])
                await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

    async def run(self):
        await self.store.init_db()

        # Start heartbeat task if URL is provided
        heartbeat_task = None
        if self.cfg.uptimerobot_url:
            heartbeat_task = asyncio.create_task(heartbeat_loop(self.cfg.uptimerobot_url, self.cfg.heartbeat_interval))

        while self.running_event.is_set():
            tasks = [self.generate_signal(sym) for sym in self.cfg.symbols]
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.cfg.poll_interval)

        if heartbeat_task:
            heartbeat_task.cancel()
        await self.store.close()
        await self.exchange.close()

    def stop(self):
        self.running_event.clear()

# -----------------------------
# Entry Point
# -----------------------------
def main():
    cfg = BotConfig()
    bot = SignalBot(cfg)

    loop = asyncio.get_event_loop()

    def shutdown_handler(*args):
        logger.info("Stopping SignalBot gracefully...")
        bot.stop()

    # Cross-platform signal handling
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig_name, shutdown_handler)
        except NotImplementedError:
            signal.signal(sig_name, lambda s, f: shutdown_handler())

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()

if __name__ == "__main__":
    main()
