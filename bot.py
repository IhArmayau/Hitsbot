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
from typing import List, Optional, Dict
from datetime import datetime, timezone
import numpy as np
import aiohttp
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, Response
import uvicorn
import re

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
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT", "UNI/USDT:USDT", "SUI/USDT:USDT"
    ])
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    confirmation_tfs: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    limit: int = int(os.getenv("LIMIT", 500))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", 300))
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.0))
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
        if self.conn:
            return
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
                outcome TEXT,
                status TEXT DEFAULT 'open'
            )
        """)
        await self.conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    async def insert_signal(self, sig: Dict):
        if not self.conn:
            await self.init_db()
        try:
            await self.conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal, entry, sl, tp, confidence, rr, outcome, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig['timestamp'], sig['symbol'], sig['signal'], float(sig['entry']),
                float(sig['sl']), float(sig['tp']), float(sig['confidence']),
                float(sig['rr']), sig.get('outcome'), 'open'
            ))
            await self.conn.commit()
            logger.info("Inserted signal into DB: %s", sig)
        except Exception as e:
            logger.exception("Failed to insert signal: %s", e)

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

def escape_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", str(text))

def format_signal_message(signals: List[Dict]) -> str:
    if not signals:
        return "_No new signals._"
    lines = ["*📊 SignalBotAI New Signals*"]
    for s in signals:
        icon = "🟢" if s['signal'] == "BUY" else "🔴"
        rr_visual = rr_bar(s.get("rr", 0.0))
        lines.append(f"{icon} *{escape_markdown(s['signal'])}* `{escape_markdown(s['symbol'])}`")
        lines.append(f"• Entry: `{s['entry']:.8f}`  SL: `{s['sl']:.8f}`  TP: `{s['tp']:.8f}`")
        lines.append(f"• R:R: `{s['rr']:.2f}` {rr_visual}  Confidence: `{s['confidence']:.2f}%`")
        lines.append(f"• Time: `{escape_markdown(s['timestamp'])}`")
        lines.append("────────")
    return "\n".join(lines)

async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.warning("Telegram send failed %s %s", resp.status, text)
        except Exception as e:
            logger.exception("Telegram error: %s", e)

def confirm_candle(df: pd.DataFrame, sig_type: str) -> bool:
    if len(df) < 3:
        return False
    prev2, prev1, curr = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if sig_type == "BUY" and curr['close'] > prev1['close'] > prev2['close'] and curr['open'] < prev1['close']:
        return True
    if sig_type == "SELL" and curr['close'] < prev1['close'] < prev2['close'] and curr['open'] > prev1['close']:
        return True
    return False

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
    if df_htf is not None and len(df_htf) > 0:
        df['htf_trend'] = 1 if df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1] else -1
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
        self.gen_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.monitor_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        self.running_event = asyncio.Event()
        self.running_event.set()
        self.model = self.load_model(cfg.ml_model_path)

    def load_model(self, path: str):
        if not path or not os.path.exists(path):
            logger.warning("ML model not found at %s. Using default probability 0.5.", path)
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded ML model from %s", path)
        return model

    def model_predict_prob(self, row: pd.Series) -> float:
        if self.model is None:
            return 0.5
        try:
            vals = [float(row.get(f, 0) or 0) for f in FEATURE_LIST]
            arr = np.array(vals).reshape(1, -1)
            prob = float(self.model.predict_proba(arr)[0][1])
            return prob
        except Exception as e:
            logger.exception("ML prediction error: %s", e)
            return 0.5

    async def fetch_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        for attempt in range(5):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=self.cfg.limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                logger.warning(f"Fetch {symbol} {tf} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1.5**attempt + random.random())
        return pd.DataFrame()

    async def generate_signal(self, symbol: str):
        async with self.gen_semaphore:
            try:
                if not self.store.conn:
                    await self.store.init_db()
                async with self.store.conn.execute(
                    "SELECT signal FROM signals WHERE symbol=? AND status='open'", (symbol,)
                ) as cursor:
                    existing_signals = [row[0] for row in await cursor.fetchall()]

                df = await self.fetch_candles(symbol, self.cfg.timeframe)
                if df.empty:
                    logger.warning(f"{symbol}: Empty entry candles")
                    return

                htf_trends = []
                for tf in self.cfg.confirmation_tfs:
                    df_htf = await self.fetch_candles(symbol, tf)
                    if not df_htf.empty:
                        trend = 1 if df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1] else -1
                        htf_trends.append(trend)
                trend_confirmed = sum(htf_trends) > 0

                df = add_indicators(df, self.cfg.indicators)
                if df.empty:
                    logger.warning(f"{symbol}: No indicators computed")
                    return

                last = df.iloc[-1]
                df_confirm = df.iloc[:-1]

                signal_type = None
                if last['close'] > last['ema_short'] > last['ema_medium'] and confirm_candle(df_confirm, "BUY") and trend_confirmed:
                    signal_type = "BUY"
                elif last['close'] < last['ema_short'] < last['ema_medium'] and confirm_candle(df_confirm, "SELL") and trend_confirmed:
                    signal_type = "SELL"

                if not signal_type or signal_type in existing_signals:
                    return

                prob = self.model_predict_prob(last)
                confidence = prob * 100
                if confidence < self.cfg.confidence_threshold:
                    confidence = 100.0

                atr = float(last['atr'] or 10)
                entry = float(last['close'])
                if signal_type == "BUY":
                    sl = entry - atr * self.cfg.indicators.atr_sl_mult
                    tp = entry + atr * self.cfg.indicators.atr_tp_mult
                else:
                    sl = entry + atr * self.cfg.indicators.atr_sl_mult
                    tp = entry - atr * self.cfg.indicators.atr_tp_mult

                rr = abs((tp - entry) / max(abs(entry - sl), 1e-8))
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

                await self.store.insert_signal(sig)

                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    msg = format_signal_message([sig])
                    await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

            except Exception as e:
                logger.exception(f"{symbol}: Error generating signal: {e}")

    async def _check_signal(self, sig_id: int, symbol: str, signal_type: str, entry: float, sl: float, tp: float):
        async with self.monitor_semaphore:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                last_price = float(ticker['last'])
            except Exception as e:
                logger.warning(f"{symbol}: Failed to fetch last price: {e}")
                return

            outcome = None
            if signal_type == "BUY":
                if last_price >= tp:
                    outcome = "TP"
                elif last_price <= sl:
                    outcome = "SL"
            else:
                if last_price <= tp:
                    outcome = "TP"
                elif last_price >= sl:
                    outcome = "SL"

            if outcome:
                await self.store.conn.execute(
                    "UPDATE signals SET status='closed', outcome=? WHERE id=?",
                    (outcome, sig_id)
                )
                await self.store.conn.commit()

                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    msg = f"✅ *Signal Closed:* {escape_markdown(signal_type)} `{escape_markdown(symbol)}`\n• Outcome: {outcome}\n• Entry: `{entry:.8f}`\n• SL: `{sl:.8f}`  TP: `{tp:.8f}`"
                    await send_telegram_message(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id, msg)

    async def monitor_open_signals(self):
        while self.running_event.is_set():
            try:
                if not self.store.conn:
                    await self.store.init_db()
                async with self.store.conn.execute(
                    "SELECT id, symbol, signal, entry, sl, tp FROM signals WHERE status='open'"
                ) as cursor:
                    rows = await cursor.fetchall()
                if rows:
                    tasks = [self._check_signal(*row) for row in rows]
                    await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.exception("Error in monitor_open_signals: %s", e)
            await asyncio.sleep(5)

    async def run(self):
        await self.store.init_db()
        try:
            while self.running_event.is_set():
                tasks = [self.generate_signal(s) for s in self.cfg.symbols]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(self.cfg.poll_interval)
        finally:
            await self.store.close()
            await self.exchange.close()

    def stop(self):
        self.running_event.clear()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
cfg = BotConfig()
bot = SignalBot(cfg)

async def safe_task(coro):
    try:
        await coro
    except Exception as e:
        logger.exception(f"Task error: {e}")

@app.on_event("startup")
async def startup_event():
    await bot.store.init_db()
    asyncio.create_task(safe_task(bot.run()))
    asyncio.create_task(safe_task(bot.monitor_open_signals()))

@app.on_event("shutdown")
async def shutdown_event():
    bot.stop()
    await bot.store.close()
    await bot.exchange.close()

@app.get("/")
async def root():
    return {"status": "alive"}

@app.api_route("/heartbeat", methods=["GET", "HEAD"])
async def heartbeat():
    logger.info("💓 Heartbeat ping received (GET/HEAD)")
    return Response(content='{"status":"alive"}', media_type="application/json")

@app.get("/stop")
async def stop_bot():
    bot.stop()
    return {"status": "stopping"}

@app.get("/signals")
async def get_signals(limit: int = 50):
    if not bot.store.conn:
        await bot.store.init_db()
    async with bot.store.conn.execute(
        "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
    ) as cursor:
        rows = await cursor.fetchall()
        columns = [c[0] for c in cursor.description]
    signals = [dict(zip(columns, r)) for r in rows]
    return {"signals": signals}

# -----------------------------
# -----------------------------
# Run with Uvicorn
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Ensure your file is named main.py
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        reload=True  # Optional: auto-reload during development
    )
