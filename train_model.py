#!/usr/bin/env python3
import ccxt.async_support as ccxt
import pandas as pd
import ta
import numpy as np
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import pickle

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
HTF_TIMEFRAME = os.getenv("HIGHER_TIMEFRAME", "4h")
TOTAL_CANDLES = int(os.getenv("TOTAL_CANDLES", 2000))
MODEL_PATH = os.getenv("ML_MODEL_PATH", "xgb_model.pkl")  # Save to project root

EMA_SHORT = int(os.getenv("EMA_SHORT", 9))
EMA_MEDIUM = int(os.getenv("EMA_MEDIUM", 21))
EMA_LONG = int(os.getenv("EMA_LONG", 50))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
BB_PERIOD = int(os.getenv("BB_PERIOD", 20))
BB_STD = float(os.getenv("BB_STD", 2.0))

FEATURE_LIST = ['ema_short','ema_medium','ema_long','rsi','atr','adx','bb_trend','vol_ok','htf_trend']

# -----------------------------
# Helpers
# -----------------------------
def add_indicators(df, df_htf=None):
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['ema_short'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], ATR_PERIOD).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], ATR_PERIOD).adx()

    bb = ta.volatility.BollingerBands(df['close'], BB_PERIOD, BB_STD)
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_trend'] = np.where(df['close'] > df['bb_middle'], 1, -1)

    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = (df['volume'] > df['vol_avg']).astype(int)

    if df_htf is not None and not df_htf.empty:
        df['htf_trend'] = np.where(df_htf['close'].iloc[-1] > df_htf['open'].iloc[-1], 1, -1)
    else:
        df['htf_trend'] = 0

    return df.dropna().reset_index(drop=True)

def generate_signal(row):
    if row['close'] > row['ema_short'] > row['ema_medium']:
        return "BUY"
    elif row['close'] < row['ema_short'] < row['ema_medium']:
        return "SELL"
    return None

def timeframe_to_minutes(tf: str):
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

async def fetch_historical(exchange, symbol, timeframe, total_candles):
    all_candles = []
    limit = 500
    tf_minutes = timeframe_to_minutes(timeframe)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    fetched = 0

    while fetched < total_candles:
        end_time = now - fetched * tf_minutes * 60 * 1000
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=min(limit, total_candles - fetched),
                params={"endTime": end_time}
            )
        except Exception as e:
            print(f"[WARN] Failed fetching {symbol} {timeframe}: {e}")
            break

        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        fetched += len(ohlcv)
        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# -----------------------------
# Main
# -----------------------------
async def main():
    exchange = ccxt.kucoinfutures({"enableRateLimit": True})
    all_data = []

    try:
        for symbol in SYMBOLS:
            print(f"Fetching {TOTAL_CANDLES} candles for {symbol}...")
            df = await fetch_historical(exchange, symbol, TIMEFRAME, TOTAL_CANDLES)
            df_htf = await fetch_historical(
                exchange,
                symbol,
                HTF_TIMEFRAME,
                TOTAL_CANDLES // max(1, (timeframe_to_minutes(HTF_TIMEFRAME)//timeframe_to_minutes(TIMEFRAME)))
            )
            df = add_indicators(df, df_htf)
            df['signal'] = df.apply(generate_signal, axis=1)
            df = df.dropna(subset=['signal'])
            df['symbol'] = symbol
            all_data.append(df[FEATURE_LIST + ['signal', 'symbol']])

        final_df = pd.concat(all_data, ignore_index=True)
        print(f"Total rows collected: {len(final_df)}")

        if final_df.empty:
            print("[ERROR] No data collected. Exiting.")
            return

        # -----------------------------
        # Train XGBoost
        # -----------------------------
        X = final_df[FEATURE_LIST].fillna(0)
        y = final_df['signal'].apply(lambda x: 1 if x=="BUY" else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")

    finally:
        await exchange.close()
        print("Exchange connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
