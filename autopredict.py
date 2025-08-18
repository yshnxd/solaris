# autopredict.py
import pandas as pd
from datetime import datetime
import pytz
import os

# Import functions from app.py
from app import download_price_cached, ensure_timestamp_in_manila, compute_real_next_time_now
from app import load_history, append_prediction_with_dedup, save_history, map_label_to_suggestion
from app import load_model_safe, get_feature_names, get_n_features, align_seq_df_to_names, align_tabular_row_to_names
from app import label_from_model

import numpy as np
from sklearn.preprocessing import MinMaxScaler

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
MARKET_TZ = pytz.timezone("US/Eastern")
OUTPUT_FILE = "predictions_history.csv"

def market_open_now():
    now = datetime.now(MARKET_TZ)
    return now.weekday() < 5 and 9 <= now.hour < 16

def run_predictions():
    if not market_open_now():
        print("Market closed. Skipping predictions.")
        return

    for ticker in TICKERS:
        try:
            df = download_price_cached(ticker, interval="60m", period="90d")
            if df is None or df.empty:
                continue

            # Align features minimally (reuse simple approach)
            last_price = df["Close"].iloc[-1]
            fetched_last_ts = df.index[-1]
            real_next_time = compute_real_next_time_now("60m")

            prediction_row = {
                "ticker": ticker,
                "interval": "60m",
                "predicted_at": ensure_timestamp_in_manila(datetime.now()),
                "fetched_last_ts": ensure_timestamp_in_manila(fetched_last_ts),
                "target_time": real_next_time,
                "predicted_label": "neutral",  # placeholder (to refine if models load)
                "suggestion": "Hold",
                "confidence": 0.0,
                "pred_price": float(last_price),
            }

            # TODO: you can import and run your models exactly like app.py does.
            # For now we just log a placeholder so GitHub Actions pipeline works.

            history = load_history()
            history = append_prediction_with_dedup(history, prediction_row, history_file=OUTPUT_FILE, save_history_func=save_history)
            print(f"Logged prediction for {ticker}")

        except Exception as e:
            print(f"Error for {ticker}: {e}")

if __name__ == "__main__":
    run_predictions()
