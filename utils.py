# utils.py
import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
import time
from config import get_tokens, CLIENT_ID
from fyers_apiv3 import fyersModel
import pandas as pd
import ta
from holidays import is_trading_day
from project_paths import data_path

logging.basicConfig(level=logging.INFO)

# Define IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

def load_symbol_master(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load symbol master: {e}")
        return {}

def get_fyers_instance():
    from config import load_latest_token_from_file, set_tokens

    # Attempt to use latest file-based token if current token is missing
    access, refresh = get_tokens()
    if not access:
        latest = load_latest_token_from_file()
        if latest:
            access, refresh = latest
            set_tokens(access, refresh)
        else:
            logging.warning("No valid token available.")
            return None

    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access, log_path="")

def fetch_historical_data(symbol, start_date_str, end_date_str):
    """Fetch historical data for a symbol between dates, skipping non-trading days
    Returns: tuple of (success: bool, message: str)"""
    fyers = get_fyers_instance()
    if fyers is None:
        return False, "Access token is missing or expired"

    # Convert input strings to date objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Check if any dates in the range are trading days
    trading_days = [
        single_date 
        for single_date in (
            start_date + timedelta(n) 
            for n in range((end_date - start_date).days + 1))
        if is_trading_day(single_date)
    ]

    if not trading_days:
        msg = f"No trading days between {start_date_str} and {end_date_str}"
        logging.info(msg)
        return False, msg

    # API request body
    data = {
        "symbol": symbol,
        "resolution": "5",
        "date_format": "1",
        "range_from": start_date_str,
        "range_to": end_date_str,
        "cont_flag": "0"
    }

    # Make API call
    response = fyers.history(data)

    if response.get("code") != 200 or "candles" not in response:
        error_msg = f"Failed to fetch data: {response.get('message', 'Unknown error')}"
        logging.error(error_msg)
        return False, error_msg

    candles = response["candles"]

    if not candles:
        error_msg = "No historical data found (likely due to holiday or wrong symbol)"
        logging.error(error_msg)
        return False, error_msg

    # Save to SQLite DB
    db_path = str(data_path("historical_data.db"))
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Insert or ignore duplicates
    for candle in candles:
        timestamp, open_, high, low, close, volume = candle
        c.execute("""
            INSERT OR IGNORE INTO historical_data
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, int(timestamp), open_, high, low, close, volume))

    conn.commit()
    conn.close()
    success_msg = f"Successfully saved {len(candles)} rows for {symbol} into historical_data.db"
    logging.info(success_msg)
    return True, success_msg
