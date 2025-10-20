#!/usr/bin/env python3
"""
Train AI model for SignalBot (KuCoin Futures version)
-----------------------------------------------------
- Fetches historical OHLCV data for symbols (BTCUSDT:USDT, etc.)
- Computes indicators identical to bot.py
- Generates labeled dataset
- Trains an XGBoost model to predict BUY/SELL probability
- Saves model as xgb_model.pkl
"""

from __future__ import annotations
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import asyncio
import logging
import os
import pickle
from dotenv import load_dotenv
from xgboost import XGBClassifier
from datetime import datetime

# -----------------------------
# Load env + setup logging
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")

# -----------------------------
# Config
# -----------------------------
SYMBOLS = [
    s.strip() for s in os.getenv(
        "SYMBOLS",
        "BTCUSDT:USDT,ETHUSDT:USDT,SOLUSDT:USDT,ADAUSDT:USDT,XRPUSDT:USDT,INJUSDT:USDT"
    ).split(",")
]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
LIMIT = int(os.getenv("LIMIT", 1000))
MODEL_OUTPUT = os.getenv("ML_MODEL_PATH", "xgb_model.pkl")

# Indicator settings
EMA_SHORT = int(os.getenv("EMA_SHORT", 9))
EMA_MEDIUM = int(os.getenv("EMA_MEDIUM", 21))
EMA_LONG = int(os.getenv("EMA_LONG", 50))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
BB_PERIOD = int(os.getenv("BB_PERIOD", 20))
BB_STD = float(os.getenv("BB_STD", 2.0))

# -----------------------------
# Helper functions
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add same indicators used in bot.py"""
    df['ema_short'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()

    bb = ta.volatility.BollingerBands(df['close'], BB_PERIOD, BB_STD)
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)

    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = (df['volume'] > df['vol_avg']).astype(int)

    df.dropna(inplace=True)
    return df

def generate_labels(df: pd.DataFrame, future_window: int = 3) -> pd.DataFrame:
    """
    Label candles:
    BUY = future close > current close * 1.002 (0.2% gain)
    SELL = future close < current close * 0.998 (0.2% drop)
    """
    df['future_close'] = df['close'].shift(-future_window)
    df['label'] = 0
    df.loc[df['future_close'] > df['close'] * 1.002, 'label'] = 1   # BUY
    df.loc[df['future_close'] < df['close'] * 0.998, 'label'] = 0   # SELL
    df.dropna(inplace=True)
    return df

FEATURES = [
    'ema_short', 'ema_medium', 'ema_long', 'rsi', 'atr',
    'adx', 'bb_trend', 'vol_ok'
]

# -----------------------------
# Data collection
# -----------------------------
async def fetch_symbol_data(symbol: str, exchange: ccxt.Exchange) -> pd.DataFrame:
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = compute_indicators(df)
        df = generate_labels(df)
        df['symbol'] = symbol
        return df
    except Exception as e:
        logger.error("Error fetching %s: %s", symbol, e)
        return pd.DataFrame()

async def collect_all_data() -> pd.DataFrame:
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})
    tasks = [fetch_symbol_data(s, exchange) for s in SYMBOLS]
    results = await asyncio.gather(*tasks)
    await exchange.close()
    df_all = pd.concat(results, ignore_index=True)
    logger.info("Collected %d rows total", len(df_all))
    return df_all

# -----------------------------
# Model training
# -----------------------------
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['label']

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X, y)
    logger.info("Training complete. Saving model...")
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved as %s", MODEL_OUTPUT)

# -----------------------------
# Entrypoint
# -----------------------------
async def main():
    logger.info("Starting training pipeline...")
    df = await collect_all_data()
    if df.empty:
        logger.error("No data collected. Exiting.")
        return
    train_model(df)
    logger.info("✅ Model training completed successfully at %s", datetime.utcnow().isoformat())

if __name__ == "__main__":
    asyncio.run(main())
