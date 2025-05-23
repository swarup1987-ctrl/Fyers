# auth.py

import webbrowser
import logging
import sqlite3
from datetime import datetime, timedelta, date
import threading
import time
from zoneinfo import ZoneInfo
from holidays import is_trading_day  # Import the holiday checker

from fyers_apiv3 import fyersModel
from config import CLIENT_ID, SECRET_KEY, REDIRECT_URI, RESPONSE_TYPE, GRANT_TYPE, set_tokens, save_tokens_to_file
from utils import fetch_historical_data
from project_paths import data_path

logging.basicConfig(filename="fyers_auth_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

session = fyersModel.SessionModel(
    client_id=CLIENT_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type=RESPONSE_TYPE,
    grant_type=GRANT_TYPE
)

DB_PATH = str(data_path("historical_data.db"))
IST = ZoneInfo("Asia/Kolkata")

def authenticate():
    url = session.generate_authcode()
    webbrowser.open(url)
    logging.info("Opened Fyers login URL in browser.")

def generate_token(auth_code: str):
    if not auth_code:
        logging.warning("No auth code entered.")
        return False

    session.set_token(auth_code)
    response = session.generate_token()
    logging.info("Attempted to generate token.")

    if response.get("s") == "ok":
        access = response["access_token"]
        refresh = response["refresh_token"]
        set_tokens(access, refresh)
        save_tokens_to_file(access, refresh)
        logging.info("Tokens saved successfully.")
        trigger_backfill()
        return True
    else:
        logging.error(f"Token generation failed: {response.get('message')}")
        return False

def refresh_token():
    response = session.refresh_token()
    if response.get("s") == "ok":
        access = response["access_token"]
        refresh = response["refresh_token"]
        set_tokens(access, refresh)
        save_tokens_to_file(access, refresh)
        logging.info("Token refreshed successfully.")
        return True
    else:
        logging.error("Token refresh failed.")
        return False

def trigger_backfill():
    """Backfill missing data, skipping non-trading days"""
    try:
        # Step 1: Get the symbol from DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data LIMIT 1")
        result = cursor.fetchone()
        conn.close()

        if not result:
            logging.warning("No existing symbol found in DB for backfill.")
            return

        symbol = result[0]

        # Step 2: Get latest timestamp for that symbol
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM historical_data WHERE symbol = ?", (symbol,))
        max_ts = cursor.fetchone()[0]
        conn.close()

        if not max_ts:
            logging.warning(f"No data found for symbol {symbol}.")
            return

        start_date = datetime.utcfromtimestamp(int(max_ts)).date() + timedelta(days=1)
        end_date = date.today()

        if start_date > end_date:
            logging.info(f"No new data to backfill for {symbol}.")
            return

        logging.info(f"Triggering one-time backfill from {start_date} to {end_date} for {symbol}.")

        current_date = start_date
        while current_date <= end_date:
            if is_trading_day(current_date):  # Use centralized holiday check
                try:
                    fetch_historical_data(
                        symbol,
                        current_date.strftime("%Y-%m-%d"),
                        current_date.strftime("%Y-%m-%d")
                    )
                except Exception as e:
                    logging.error(f"Backfill failed for {symbol} on {current_date}: {e}")
            else:
                logging.info(f"Skipped non-trading day: {current_date}")
            
            current_date += timedelta(days=1)

    except Exception as e:
        logging.error(f"Backfill process failed: {e}")

def start_scheduler():
    import schedule

    schedule.every(1).hour.do(refresh_token)

    def run():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=run, daemon=True).start()
    logging.info("Started background scheduler.")
