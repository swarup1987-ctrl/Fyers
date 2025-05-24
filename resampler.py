import sqlite3
from datetime import datetime, time, timedelta
from project_paths import data_path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

resampled_data = []          # Holds 5 min data as list of dicts
resampled_15min = []         # Holds 15 min resampled data as list of dicts
resampled_1hour = []         # Holds 1 hour resampled data as list of dicts

def fetch_and_store_symbol(symbol, db_path=None):
    """
    Fetches historical data for a given symbol from the SQLite DB and stores it in memory.
    """
    global resampled_data, resampled_15min, resampled_1hour
    resampled_data = []
    resampled_15min = []
    resampled_1hour = []
    if db_path is None:
        db_path = str(data_path("historical_data.db"))
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, symbol, timestamp, open, high, low, close, volume 
            FROM historical_data 
            WHERE UPPER(symbol) = ?
            ORDER BY timestamp ASC
        """, (symbol.strip().upper(),))
        rows = cur.fetchall()
        conn.close()
        if rows:
            resampled_data = [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "timestamp": row[2],
                    "open": row[3],
                    "high": row[4],
                    "low": row[5],
                    "close": row[6],
                    "volume": row[7]
                }
                for row in rows
            ]
            resample_all()
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in fetch_and_store_symbol: {e}")
        return False

def get_resampled_data(interval="5min"):
    if interval == "5min":
        return resampled_data
    elif interval == "15min":
        return resampled_15min
    elif interval == "1hour":
        return resampled_1hour
    else:
        return []

def resample_all():
    """Resample the 5 min data to 15 min and 1 hour data."""
    global resampled_15min, resampled_1hour
    resampled_15min = resample(resampled_data, 15)
    resampled_1hour = resample(resampled_data, 60)

def resample(data, interval_minutes):
    """
    data: list of dicts (must have keys: timestamp, open, high, low, close, volume)
    interval_minutes: int (e.g., 15 or 60)
    Returns: list of dicts (same keys as input, timestamp is start of resampled interval)
    """
    if not data:
        return []
    # Convert data to DataFrame for easier handling
    import pandas as pd
    df = pd.DataFrame(data)
    if df.empty:
        return []
    # Explicitly cast 'timestamp' to int to avoid pandas FutureWarning
    df['timestamp'] = df['timestamp'].astype(int)
    # Convert 'timestamp' to UTC datetime and then to IST
    df['dt_ist'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True).dt.tz_convert(IST)
    # Group data by trading day (in IST)
    out = []
    for trade_date, day_df in df.groupby(df['dt_ist'].dt.date):
        session_start = datetime.combine(trade_date, time(9, 15), IST)
        session_end = datetime.combine(trade_date, time(15, 25), IST)
        bucket_start = session_start
        while bucket_start < session_end:
            bucket_end = bucket_start + timedelta(minutes=interval_minutes)
            if bucket_end > session_end + timedelta(minutes=1):  # ensure we don't overshoot last interval
                break
            mask = (day_df['dt_ist'] >= bucket_start) & (day_df['dt_ist'] < bucket_end)
            bucket = day_df[mask]
            if not bucket.empty:
                # Aggregate the bucket
                try:
                    utc_bucket_start = bucket_start.astimezone(ZoneInfo("UTC"))
                except Exception:
                    import pytz
                    utc_bucket_start = bucket_start.astimezone(pytz.UTC)
                out.append({
                    "id": int(bucket.iloc[0].get("id", 0)),
                    "symbol": bucket.iloc[0]["symbol"],
                    "timestamp": int(utc_bucket_start.timestamp()),
                    "open": float(bucket.iloc[0]["open"]),
                    "high": float(bucket["high"].max()),
                    "low": float(bucket["low"].min()),
                    "close": float(bucket.iloc[-1]["close"]),
                    "volume": float(bucket["volume"].sum()),
                })
            bucket_start += timedelta(minutes=interval_minutes)
    return out
