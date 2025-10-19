
from __future__ import annotations
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import ta
import logging
import os
import sqlite3
import json
import tempfile
import time
import math
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, jsonify, Response
import aiohttp
import aiofiles
import signal
from concurrent.futures import ThreadPoolExecutor

# Optional prometheus
try:
    from prometheus_client import Summary, Gauge, Counter, generate_latest, CollectorRegistry
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# LOGGING (structured)
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("professional_signal_bot")

# -----------------------------
# CONFIG DATACLASS
# -----------------------------
@dataclass
class IndicatorsConfig:
    ema_short: int = 9
    ema_long: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0

@dataclass
class TelegramConfig:
    bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    rate_limit: float = float(os.getenv("TELEGRAM_RATE_LIMIT", "1.0"))

@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "INJ/USDT:USDT",
        "ADA/USDT:USDT",
        "XPL/USDT:USDT",
        "XRP/USDT:USDT"
    ])
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    limit: int = int(os.getenv("LIMIT", "100"))
    poll_interval_sec: int = int(os.getenv("POLL_INTERVAL_SEC", "600"))
    csv_file: str = os.getenv("CSV_FILE", "signals.csv")
    csv_save_batch: int = int(os.getenv("CSV_SAVE_BATCH", "5"))
    max_csv_signals: int = int(os.getenv("MAX_CSV_SIGNALS", "1000"))
    smc_lookback_structure: int = int(os.getenv("SMC_LOOKBACK", "5"))
    smc_threshold: float = float(os.getenv("SMC_THRESHOLD", "0.8"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "50"))
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    sqlite_db: str = os.getenv("SQLITE_DB", "signals.db")
    max_concurrent_fetches: int = int(os.getenv("MAX_CONCURRENT_FETCHES", "5"))
    per_symbol_rate_limit_sec: float = float(os.getenv("PER_SYMBOL_RATE_LIMIT_SEC", "0.5"))


# -----------------------------
# METRICS (optional)
# -----------------------------
if PROM_AVAILABLE:
    REG = CollectorRegistry()
    METRIC_FETCH_DURATION = Summary("fetch_ohlcv_seconds", "Time taken to fetch OHLCV", registry=REG)
    METRIC_SIGNALS_GENERATED = Counter("signals_generated_total", "Total number of signals generated", registry=REG)
    METRIC_TELEGRAM_SENT = Counter("telegram_sent_total", "Total telegram messages sent", registry=REG)
    METRIC_FETCH_ERRORS = Counter("fetch_errors_total", "Total fetch errors", registry=REG)
else:
    REG = None
    METRIC_FETCH_DURATION = METRIC_SIGNALS_GENERATED = METRIC_TELEGRAM_SENT = METRIC_FETCH_ERRORS = None


# -----------------------------
# UTILITIES
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def escape_markdown_v2(text: str) -> str:
    # Telegram MarkdownV2 escaping
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([{}])'.format(re.escape(escape_chars)), r'\\\1', text)

def backoff_sleep(attempt: int):
    # exponential backoff with jitter
    base = min(60, (2 ** attempt))
    jitter = base * 0.1
    return base - jitter + (jitter * 2 * (os.urandom(1)[0] / 255.0))

# -----------------------------
# PERSISTENCE (SQLite via run_in_executor)
# -----------------------------
class SignalStore:
    """Simple SQLite-backed store for signals and last_signal tracking."""
    def __init__(self, db_path: str, executor: ThreadPoolExecutor):
        self.db_path = db_path
        self.executor = executor
        self._initialized = False

    def _init_db_sync(self):
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TIMESTAMP,
            symbol TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            candle_time TIMESTAMP,
            confidence REAL,
            ema_trend REAL,
            rsi REAL,
            rr REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS last_signal (
            symbol TEXT PRIMARY KEY,
            payload TEXT,
            updated TIMESTAMP
        )
        """)
        con.commit()
        con.close()
        self._initialized = True

    async def init(self):
        if not self._initialized:
            await asyncio.get_running_loop().run_in_executor(self.executor, self._init_db_sync)

    async def upsert_last_signal(self, symbol: str, payload: dict):
        def sync_upsert():
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("""
                INSERT INTO last_signal (symbol, payload, updated) VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET payload=excluded.payload, updated=excluded.updated
            """, (symbol, json.dumps(payload, default=str), datetime.utcnow()))
            con.commit()
            con.close()
        await asyncio.get_running_loop().run_in_executor(self.executor, sync_upsert)

    async def get_all_last_signals(self) -> List[dict]:
        def sync_get():
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("SELECT payload FROM last_signal")
            rows = cur.fetchall()
            con.close()
            return [json.loads(r[0]) for r in rows if r[0]]
        return await asyncio.get_running_loop().run_in_executor(self.executor, sync_get)

    async def insert_signal(self, signal: dict):
        def sync_insert():
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("""
                INSERT INTO signals (time, symbol, signal, entry, sl, tp, candle_time, confidence, ema_trend, rsi, rr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("time"),
                signal.get("symbol"),
                signal.get("signal"),
                signal.get("entry"),
                signal.get("sl"),
                signal.get("tp"),
                signal.get("candle_time"),
                signal.get("confidence"),
                signal.get("ema_trend"),
                signal.get("rsi"),
                signal.get("rr")
            ))
            con.commit()
            con.close()
        await asyncio.get_running_loop().run_in_executor(self.executor, sync_insert)

# -----------------------------
# PROFESSIONAL SIGNAL BOT
# -----------------------------
class ProfessionalSignalBot:
    def __init__(self, cfg: BotConfig):
        self.config = cfg
        self.exchange = ccxt.kucoinfutures({"enableRateLimit": True})
        self.ohlcv_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.fetch_semaphore = asyncio.Semaphore(self.config.max_concurrent_fetches)
        self.telegram_queue: asyncio.Queue[str] = asyncio.Queue()
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.store = SignalStore(self.config.sqlite_db, self.executor)
        self.local_signals_df = pd.DataFrame(columns=[
            'time', 'symbol', 'signal', 'entry', 'sl', 'tp', 'candle_time', 'confidence', 'ema_trend', 'rsi', 'rr'
        ])
        # per-symbol last-send timestamp to respect rate limits
        self._symbol_last_sent: Dict[str, float] = {}
        # tasks
        self._tasks: List[asyncio.Task] = []
        # warm-up flag
        self._markets_loaded = False

    async def start(self):
        logger.info("Starting ProfessionalSignalBot")
        await self.store.init()
        if not self.session:
            self.session = aiohttp.ClientSession()
        # warm load markets (best-effort)
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True
            logger.debug("Loaded exchange markets")
        except Exception as e:
            logger.warning(f"Failed to load markets: {e}")
        # start background workers
        self._tasks.append(asyncio.create_task(self._telegram_worker()))
        self._tasks.append(asyncio.create_task(self._main_loop()))

    async def stop(self):
        logger.info("Stopping ProfessionalSignalBot")
        self.running = False
        # wait for tasks to finish cleanly
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # flush buffered signals to DB & CSV
        await self._flush_signals()
        # close session & exchange
        if self.session:
            await self.session.close()
        try:
            await self.exchange.close()
        except Exception:
            pass
        self.executor.shutdown(wait=True)

    # -----------------------------
    # TELEGRAM WORKER
    # -----------------------------
    async def send_telegram(self, message: str):
        # public queuing method
        await self.telegram_queue.put(message)

    async def _telegram_worker(self):
        logger.info("Telegram worker started")
        cfg = self.config.telegram
        while self.running or not self.telegram_queue.empty():
            try:
                message = await self.telegram_queue.get()
            except asyncio.CancelledError:
                break
            if not cfg.bot_token or not cfg.chat_id:
                logger.debug("Telegram not configured; skipping message")
                self.telegram_queue.task_done()
                continue

            # respect global rate limit and per-symbol rate limit (approx)
            # simple throttling: ensure X seconds since last telegram
            sleep_time = 0
            now = time.time()
            if now - getattr(self, "last_telegram_time", 0) < cfg.rate_limit:
                sleep_time = cfg.rate_limit - (now - getattr(self, "last_telegram_time", 0))

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
            payload = {
                "chat_id": cfg.chat_id,
                "text": escape_markdown_v2(message),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            success = False
            for attempt in range(3):
                try:
                    if not self.session:
                        self.session = aiohttp.ClientSession()
                    async with self.session.post(url, json=payload, timeout=10) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            success = True
                            setattr(self, "last_telegram_time", time.time())
                            if PROM_AVAILABLE:
                                METRIC_TELEGRAM_SENT.inc()
                            break
                        else:
                            logger.warning(f"Telegram send failed: status={resp.status} body={text}")
                except Exception as e:
                    logger.warning(f"Telegram send attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
            if not success:
                logger.error("Telegram send failed after retries")
            self.telegram_queue.task_done()
        logger.info("Telegram worker stopped")

    # -----------------------------
    # FETCH OHLCV (cached + backoff)
    # -----------------------------
    async def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        key = (symbol, self.config.timeframe)
        now_ts = time.time()
        cache = self.ohlcv_cache.get(key)
        # cache TTL: 300 seconds
        if cache and (now_ts - cache[0]) < 300:
            return cache[1]

        for attempt in range(4):
            try:
                async with self.fetch_semaphore:
                    t0 = time.time()
                    raw = await self.exchange.fetch_ohlcv(symbol, timeframe=self.config.timeframe, limit=self.config.limit)
                    duration = time.time() - t0
                    if PROM_AVAILABLE:
                        METRIC_FETCH_DURATION.observe(duration)
                    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
                    df["time"] = pd.to_datetime(df["time"], unit='ms', utc=True)
                    df.set_index("time", inplace=True)
                    df = self._add_indicators(df)
                    self.ohlcv_cache[key] = (time.time(), df)
                    return df
            except Exception as e:
                logger.warning(f"fetch_ohlcv({symbol}) attempt {attempt+1} failed: {e}")
                if PROM_AVAILABLE:
                    METRIC_FETCH_ERRORS.inc()
                await asyncio.sleep(backoff_sleep(attempt))
        logger.error(f"Failed to fetch OHLCV for {symbol} after retries")
        return pd.DataFrame()

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        ind = self.config.indicators
        df['ema_short'] = df['close'].ewm(span=ind.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ind.ema_long, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=ind.ema_trend, adjust=False).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], ind.rsi_period).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ind.atr_period).average_true_range()

        # dynamic rsi bands (guard zero-divide)
        close_mean = df['close'].rolling(14).mean().replace(0, pd.NA)
        atr_mean = df['atr'].rolling(14).mean().replace(0, pd.NA)
        ratio = (atr_mean / close_mean).fillna(0)
        df['rsi_overbought_dyn'] = 70 + (ratio * 50)
        df['rsi_oversold_dyn'] = 30 - (ratio * 50)
        df = df.dropna(subset=['ema_short', 'ema_long', 'ema_trend', 'rsi', 'atr'])
        return df

    # -----------------------------
    # SIGNAL GENERATION (keeps your logic)
    # -----------------------------
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[dict]:
        if df.empty or len(df) < 2:
            return None

        latest = df.iloc[-1]
        ind = self.config.indicators
        trend = "BUY" if latest["ema_short"] > latest["ema_long"] else "SELL"

        rsi_overbought = float(latest.get('rsi_overbought_dyn', ind.rsi_overbought))
        rsi_oversold = float(latest.get('rsi_oversold_dyn', ind.rsi_oversold))
        if trend == "BUY" and latest["rsi"] > rsi_overbought:
            return None
        if trend == "SELL" and latest["rsi"] < rsi_oversold:
            return None

        lookback = self.config.smc_lookback_structure
        if len(df) < lookback + 1:
            return None

        highs = df['high'].iloc[-(lookback+1):].reset_index(drop=True)
        lows = df['low'].iloc[-(lookback+1):].reset_index(drop=True)
        match_count = 0
        for i in range(1, len(highs)):
            if trend == "BUY":
                if highs.iloc[i] > highs.iloc[i-1] and lows.iloc[i] > lows.iloc[i-1]:
                    match_count += 1
            else:
                if lows.iloc[i] < lows.iloc[i-1] and highs.iloc[i] < highs.iloc[i-1]:
                    match_count += 1

        if (match_count / lookback) < self.config.smc_threshold:
            return None

        close_mean_14 = df['close'].rolling(14).mean().iloc[-1] if len(df) >= 14 else df['close'].mean()
        atr_val = float(latest['atr'])
        atr_ratio = (atr_val / close_mean_14) if close_mean_14 and not math.isnan(close_mean_14) else 0.0

        sl_distance = atr_val * ind.atr_sl_mult * (1 + atr_ratio)
        tp_distance = atr_val * ind.atr_tp_mult * (1 + atr_ratio)
        entry = float(latest['close'])
        if trend == "BUY":
            sl = entry - sl_distance
            tp = entry + tp_distance
            rr = (tp - entry) / abs(entry - sl) if sl != entry else float('nan')
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
            rr = (entry - tp) / abs(sl - entry) if sl != entry else float('nan')

        # confidence metric
        try:
            price_diff_pct = abs(latest['ema_short'] - latest['ema_long']) / latest['close'] * 100 * 10
            rsi_bonus = 50 if (trend == "BUY" and latest['rsi'] > 50) or (trend == "SELL" and latest['rsi'] < 50) else 0
            confidence = float(min(max(price_diff_pct + rsi_bonus, 0), 100))
        except Exception:
            confidence = 0.0

        if confidence < self.config.confidence_threshold:
            return None

        signal = {
            "time": now_utc().isoformat(),
            "symbol": symbol,
            "signal": trend,
            "entry": round(entry, 8),
            "sl": round(sl, 8),
            "tp": round(tp, 8),
            "confidence": round(confidence, 2),
            "ema_trend": round(float(latest['ema_trend']), 8),
            "rsi": round(float(latest['rsi']), 2),
            "candle_time": df.index[-1].to_pydatetime().isoformat(),
            "rr": round(rr, 2) if not math.isnan(rr) else None
        }

        # duplicate suppression using store last signals
        last = None
        # we keep an in-memory last-signal map (persisted in DB)
        last_signal = next((s for s in self.local_signals_df.to_dict('records') if s.get('symbol') == symbol), None)
        if last_signal:
            last = last_signal
        if last:
            same_direction = last['signal'] == signal['signal']
            small_price_diff = abs(float(last['entry']) - signal['entry']) < 0.001 * signal['entry']
            similar_confidence = abs(float(last['confidence']) - signal['confidence']) < 1
            if same_direction and small_price_diff and similar_confidence:
                return None

        return signal

    # -----------------------------
    # PERSISTENCE + CSV flush
    # -----------------------------
    async def _flush_signals(self):
        logger.info("Flushing signals to store & CSV")
        # persist DataFrame rows to sqlite and CSV
        rows = self.local_signals_df.to_dict('records')
        if not rows:
            return
        for row in rows:
            try:
                await self.store.insert_signal(row)
                await self.store.upsert_last_signal(row['symbol'], row)
            except Exception as e:
                logger.exception(f"Failed to persist row {row}: {e}")

        # write CSV atomically
        csv_text = self.local_signals_df.to_csv(index=False)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp")
        os.close(tmp_fd)
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                await f.write(csv_text)
            os.replace(tmp_path, self.config.csv_file)
            logger.info(f"Wrote CSV to {self.config.csv_file}")
        except Exception as e:
            logger.exception(f"Failed to write CSV: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        # clear local buffer
        self.local_signals_df = self.local_signals_df.iloc[0:0]

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    async def _main_loop(self):
        logger.info("Main loop started")
        poll = self.config.poll_interval_sec
        symbols = list(self.config.symbols)
        # pre-populate last signals from DB
        try:
            last_signals = await self.store.get_all_last_signals()
            for s in last_signals:
                try:
                    self.local_signals_df = pd.concat([self.local_signals_df, pd.DataFrame([s])], ignore_index=True)
                except Exception:
                    pass
        except Exception:
            pass

        while self.running:
            start = time.time()
            symbol_tasks = []
            for symbol in symbols:
                symbol_tasks.append(asyncio.create_task(self._process_symbol(symbol)))

            # wait for all symbol processing
            try:
                await asyncio.gather(*symbol_tasks)
            except Exception as e:
                logger.exception(f"Error while gathering symbol tasks: {e}")

            # flush buffer periodically or when size reached
            if len(self.local_signals_df) >= self.config.csv_save_batch:
                await self._flush_signals()

            # ensure fixed poll interval
            elapsed = time.time() - start
            to_sleep = max(0.0, poll - elapsed)
            # sleep in small chunks so we can shutdown quickly
            slept = 0.0
            while self.running and slept < to_sleep:
                await asyncio.sleep(min(1.0, to_sleep - slept))
                slept += 1.0

        # final flush on exit
        await self._flush_signals()
        logger.info("Main loop stopped")

    async def _process_symbol(self, symbol: str):
        # per-symbol rate limiting
        last_sent = self._symbol_last_sent.get(symbol, 0.0)
        now = time.time()
        if now - last_sent < self.config.per_symbol_rate_limit_sec:
            await asyncio.sleep(self.config.per_symbol_rate_limit_sec - (now - last_sent))

        df = await self.fetch_ohlcv(symbol)
        if df.empty:
            return

        signal = self.generate_signal(df, symbol)
        if signal:
            logger.info(f"Signal generated: {signal['symbol']} {signal['signal']} entry={signal['entry']} conf={signal['confidence']}")
            # add to local buffer
            self.local_signals_df = pd.concat([self.local_signals_df, pd.DataFrame([signal])], ignore_index=True)
            # persist and update last_signal in DB asynchronously
            try:
                await self.store.insert_signal(signal)
                await self.store.upsert_last_signal(symbol, signal)
            except Exception as e:
                logger.exception(f"Failed to persist generated signal: {e}")

            # prepare telegram message
            msg = (f"*{signal['symbol']}* {signal['signal']} @ {signal['entry']}\n"
                   f"SL: {signal['sl']} | TP: {signal['tp']} | Confidence: {signal['confidence']} | R/R: {signal['rr']}")
            await self.send_telegram(msg)
            self._symbol_last_sent[symbol] = time.time()
            if PROM_AVAILABLE:
                METRIC_SIGNALS_GENERATED.inc()

    # -----------------------------
    # UTIL: Health / Status
    # -----------------------------
    def health_status(self) -> dict:
        return {
            "running": self.running,
            "symbols": self.config.symbols,
            "last_signals_buffered": len(self.local_signals_df),
            "last_fetch_cache_keys": len(self.ohlcv_cache),
            "utc_time": now_utc().isoformat()
        }


# -----------------------------
# FLASK HTTP (health & metrics)
# -----------------------------
app = Flask(__name__)
BOT: Optional[ProfessionalSignalBot] = None

@app.route("/")
def home():
    return "Professional KuCoin Perpetual Signal Bot is running!", 200

@app.route("/status")
def status():
    if not BOT:
        return jsonify({"error": "bot not started"}), 500
    return jsonify(BOT.health_status())

@app.route("/heartbeat")
def heartbeat():
    logger.info("Heartbeat ping")
    return "Alive", 200

@app.route("/metrics")
def metrics():
    if PROM_AVAILABLE and REG:
        data = generate_latest(REG)
        return Response(data, mimetype="text/plain; version=0.0.4; charset=utf-8")
    return Response("prometheus_client not installed", status=501)

# -----------------------------
# RUN & SIGNAL HANDLING
# -----------------------------
async def run_bot():
    global BOT
    cfg = BotConfig()
    BOT = ProfessionalSignalBot(cfg)
    await BOT.start()

    # run forever until stopped
    while BOT.running:
        await asyncio.sleep(1)

def _start_flask_in_thread():
    port = int(os.getenv("PORT", "10000"))
    from threading import Thread
    Thread(target=lambda: app.run(host="0.0.0.0", port=port, use_reloader=False), daemon=True).start()

def main():
    _start_flask_in_thread()
    loop = asyncio.get_event_loop()

    main_task = loop.create_task(run_bot())

    def _on_exit(signum, frame):
        logger.info(f"Received signal {signum}, shutting down.")
        if BOT:
            loop.create_task(BOT.stop())

    # handle SIGINT/SIGTERM
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _on_exit)

    try:
        loop.run_until_complete(main_task)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Main loop interrupted")
    finally:
        # attempt final stop
        if BOT:
            loop.run_until_complete(BOT.stop())
        loop.close()
        logger.info("Exit complete")

if __name__ == "__main__":
    main()
