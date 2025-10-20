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
from fastapi import FastAPI
import uvicorn

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SignalBotAI")

# -----------------------------
# Configs
# -----------------------------
@dataclass
class IndicatorsConfig:
    ema_short: int = int(os.getenv("EMA_SHORT", 9))
    ema_medium: int = int(os.getenv("EMA_MEDIUM", 21))
    ema_long: int = int(os.getenv("EMA_LONG", 50))
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    atr_period: int = int(os.getenv("ATR_PERIOD", 14))
    bb_period: int = int(os.getenv("BB_PERIOD", 20))
    bb_std: float = float(os.getenv("BB_STD", 2.0))
    atr_tp_mult: float = float(os.getenv("ATR_TP_MULT", 3.0))
    atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", 1.5))

@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: [
        s.strip() for s in os.getenv(
            "SYMBOLS",
            "BTCUSDT:USDT,ETHUSDT:USDT,SOLUSDT:USDT,ADAUSDT:USDT,XRPUSDT:USDT,XPLUSDT:USDT,INJUSDT:USDT"
        ).split(",")
    ])
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "4h")
    limit: int = int(os.getenv("LIMIT", 500))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 70.0))
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "xgb_model.pkl")
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

# -----------------------------
# Database
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
        logger.info("Database initialized at %s", self.db_path)

    async def insert_signal(self, sig: Dict):
        if not self.conn:
            logger.warning("DB not initialized")
            return
        try:
            await self.conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal, entry, sl, tp, confidence, rr, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig['timestamp'], sig['symbol'], sig['signal'], float(sig['entry']),
                float(sig['sl']), float(sig['tp']), float(sig['confidence']),
                float(sig['rr']), sig.get('outcome')
            ))
            await self.conn.commit()
            logger.info("Inserted signal into DB: %s", sig)
        except Exception as e:
            logger.error("DB insert failed: %s", e)

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

# -----------------------------
# Helpers
# -----------------------------
def rr_bar(rr_value: float, max_rr: float = 5.0, length: int = 10) -> str:
    blocks = min(int((rr_value / max_rr) * length), length)
    return "🟩" * max(0, blocks) + "⬜" * (length - max(0, blocks))

def format_signal_message(signals: List[Dict]) -> str:
    if not signals:
        return "_No new signals._"
    lines = ["*📊 SignalBotAI New Signals*"]
    for s in signals:
        icon = "🟢" if s['signal'] == "BUY" else "🔴"
        rr_visual = rr_bar(s.get("rr", 0.0))
        lines.append(f"{icon} *{s['signal']}* `{s['symbol']}`")
        lines.append(f"• Entry: `{s['entry']:.8f}`  SL: `{s['sl']:.8f}`  TP: `{s['tp']:.8f}`")
        lines.append(f"• R:R: `{s['rr']:.2f}` {rr_visual}  Confidence: `{s['confidence']:.2f}%`")
        lines.append("────────")
    return "\n".join(lines)

async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.warning("Telegram send failed %s %s", resp.status, text)
        except Exception as e:
            logger.error("Telegram error: %s", e)

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
# Feature Engineering
# -----------------------------
FEATURE_LIST = ['ema_short','ema_medium','ema_long','rsi','atr','adx','bb_trend','vol_ok','htf_trend']

def add_indicators(df: pd.DataFrame, ind_cfg: IndicatorsConfig, df_htf: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ema_short'] = df['close'].ewm(span=ind_cfg.ema_short, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=ind_cfg.ema_medium, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ind_cfg.ema_long, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind_cfg.rsi_period).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind_cfg.atr_period).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ind_cfg.atr_period).adx()
    bb = ta.volatility.BollingerBands(df['close'], ind_cfg.bb_period, ind_cfg.bb_std)
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = (df['volume'] > df['vol_avg']).astype(int)
    if df_htf is not None and not df_htf.empty:
        df['htf_trend'] = np.where(df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1], 1, -1)
    else:
        df['htf_trend'] = 0
    return df.dropna().reset_index(drop=True)

# -----------------------------
# SignalBot
# -----------------------------
class SignalBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.exchange = ccxt.kucoinfutures({"enableRateLimit": True})
        self.store = SignalStore(cfg.sqlite_db)
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.running_event = asyncio.Event()
        self.running_event.set()
        self.model = self.load_model(cfg.ml_model_path)

    def load_model(self, path: str):
        if not path or not os.path.exists(path):
            logger.warning("ML model not found at %s. Running without AI predictions.", path)
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded ML model from %s", path)
        return model

    def model_predict_prob(self, row: pd.Series) -> float:
        if self.model is None:
            logger.info("ML model not loaded. Returning default probability 0.5")
            return 0.5

        try:
            vals = []
            for f in FEATURE_LIST:
                val = row.get(f, 0)
                if pd.isna(val):
                    val = 0.0
                vals.append(float(val))

            arr = np.array(vals).reshape(1, -1)
            prob = float(self.model.predict_proba(arr)[0][1])

            logger.info(
                "ML prediction for %s: %.2f%% (features: %s)",
                row.get("symbol", "unknown"),
                prob * 100,
                {f: row.get(f, 0) for f in FEATURE_LIST}
            )
            return prob

        except Exception as e:
            logger.error("Error predicting ML probability: %s", e)
            return 0.5

    async def fetch_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=self.cfg.limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                logger.warning("Fetch %s %s attempt %d failed: %s", symbol, tf, attempt+1, e)
                await asyncio.sleep(1.5**attempt + random.random())
        return pd.DataFrame()

    async def generate_signal(self, symbol: str):
        async with self.semaphore:
            logger.info("Fetching candles for %s", symbol)
            df = await self.fetch_candles(symbol, self.cfg.timeframe)
            df_htf = await self.fetch_candles(symbol, self.cfg.higher_timeframe)

            if df.empty:
                logger.warning("No candles returned for %s %s", symbol, self.cfg.timeframe)
                return
            if df_htf.empty:
                logger.warning("No HTF candles returned for %s %s", symbol, self.cfg.higher_timeframe)
                return

            df = add_indicators(df, self.cfg.indicators, df_htf)
            if df.empty:
                logger.warning("Indicators resulted in empty DataFrame for %s", symbol)
                return

            last = df.iloc[-1]
            signal_type = None

            logger.info(
                "Checking signal for %s: close=%.8f, ema_short=%.8f, ema_medium=%.8f",
                symbol, last['close'], last['ema_short'], last['ema_medium']
            )

            if last['close'] > last['ema_short'] > last['ema_medium'] and confirm_candle(df, "BUY"):
                signal_type = "BUY"
            elif last['close'] < last['ema_short'] < last['ema_medium'] and confirm_candle(df, "SELL"):
                signal_type = "SELL"

            if not signal_type:
                logger.info("No signal generated for %s", symbol)
                return

            # ML probability
            prob = self.model_predict_prob(last)
            confidence = prob * 100.0
            logger.info("Signal type: %s, confidence: %.2f%%", signal_type, confidence)

            if confidence < self.cfg.confidence_threshold:
                logger.info(
                    "Confidence %.2f%% below threshold %.2f%% for %s. Skipping signal.",
                    confidence, self.cfg.confidence_threshold, symbol
                )
                return

            entry = float(last['close'])
            atr = float(last['atr'] or 0)
            if atr <= 0:
                logger.warning("ATR <= 0 for %s. Skipping signal.", symbol)
                return

            if signal_type == "BUY":
                sl = entry - atr * self.cfg.indicators.atr_sl_mult
                tp = entry + atr * self.cfg.indicators.atr_tp_mult
            else:
                sl = entry + atr * self.cfg.indicators.atr_sl_mult
                tp = entry - atr * self.cfg.indicators.atr_tp_mult

            rr = abs((tp - entry) / abs(entry - sl)) if entry - sl != 0 else 0.0

            sig = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal": signal_type,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
                "rr": rr
            }

            logger.info("Generated signal for %s: %s", symbol, sig)
            await self.store.insert_signal(sig)

            if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                msg = format_signal_message([sig])
                await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

    async def run(self):
        logger.info("Initializing database...")
        await self.store.init_db()
        logger.info("Database initialized. Starting main loop...")

        try:
            while self.running_event.is_set():
                logger.info("Starting new poll cycle for symbols: %s", self.cfg.symbols)
                tasks = [self.generate_signal(s) for s in self.cfg.symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error("Error in task for symbol %s: %s", self.cfg.symbols[idx], res)

                logger.info("Poll cycle complete. Sleeping for %d seconds.", self.cfg.poll_interval)
                await asyncio.sleep(self.cfg.poll_interval)

        except Exception as e:
            logger.error("Unexpected error in run loop: %s", e)
        finally:
            logger.info("Shutting down bot...")
            await self.store.close()
            await self.exchange.close()
            logger.info("Bot stopped cleanly.")

    def stop(self):
        self.running_event.clear()
        logger.info("Stop signal received for bot.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
cfg = BotConfig()
bot = SignalBot(cfg)

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting SignalBotAI...")
        await bot.store.init_db()
        logger.info("Database initialized successfully.")
        asyncio.create_task(bot.run())
        logger.info("SignalBot main loop started.")
    except Exception as e:
        logger.error("Error during startup: %s", e)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down SignalBotAI...")
    bot.stop()
    await bot.store.close()
    await bot.exchange.close()
    logger.info("Shutdown complete.")

@app.get("/")
async def root():
    return {"status": "alive"}

@app.get("/heartbeat")
async def heartbeat():
    logger.info("💓 Heartbeat ping received")
    return {"status": "alive"}

@app.get("/stop")
async def stop_bot():
    bot.stop()
    return {"status": "stopping"}
