import sqlite3
from datetime import datetime, time
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
    Fast vectorized resampling using pandas.
    data: list of dicts (must have keys: timestamp, open, high, low, close, volume)
    interval_minutes: int (e.g., 15 or 60)
    Returns: list of dicts (same keys as input, timestamp is start of resampled interval)
    """
    if not data:
        return []
    import pandas as pd
    df = pd.DataFrame(data)
    if df.empty:
        return []
    df['timestamp'] = df['timestamp'].astype(int)
    # Convert to IST
    df['dt_ist'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(IST)
    df = df.set_index('dt_ist')
    rule = f'{interval_minutes}min'
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'id': 'first',
        'symbol': 'first',
        'timestamp': 'first',  # We'll overwrite this below
    }
    result = df.resample(rule, label='left', closed='left').agg(agg)
    # Filter only session bars (09:15 to 15:25 IST)
    session_start = time(9, 15)
    session_end = time(15, 25)
    result = result[(result.index.time >= session_start) & (result.index.time <= session_end)]
    # Overwrite 'timestamp' with UTC timestamp of bucket start
    result['timestamp'] = result.index.tz_convert('UTC').astype(int) // 10**9
    # Drop buckets where open/close/high/low are NaN (no data in bucket)
    result = result.dropna(subset=['open', 'close', 'high', 'low'])
    out = result.reset_index(drop=True).to_dict(orient='records')
    return out
