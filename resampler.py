import sqlite3
from datetime import datetime, timedelta
from project_paths import data_path

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
    # Prepare: ensure data is sorted by timestamp
    data = sorted(data, key=lambda x: x['timestamp'])
    out = []
    bucket = []
    current_start = None
    for row in data:
        # Convert timestamp to datetime if not already
        ts = row['timestamp']
        if isinstance(ts, str):
            ts = int(ts)
        dt = datetime.fromtimestamp(ts)
        # Compute interval start for this row
        interval_start = dt.replace(minute=(dt.minute // interval_minutes) * interval_minutes, second=0, microsecond=0)
        if current_start is None:
            current_start = interval_start
        if interval_start != current_start and bucket:
            out.append(aggregate_bucket(bucket, current_start))
            bucket = []
            current_start = interval_start
        bucket.append(row)
    # Last bucket
    if bucket:
        out.append(aggregate_bucket(bucket, current_start))
    return out

def aggregate_bucket(bucket, interval_start):
    """Aggregate a list of OHLCV rows into a single OHLCV row for the bucket."""
    return {
        "id": bucket[0].get("id", None),
        "symbol": bucket[0]["symbol"],
        "timestamp": int(interval_start.timestamp()),
        "open": bucket[0]["open"],
        "high": max(row["high"] for row in bucket),
        "low": min(row["low"] for row in bucket),
        "close": bucket[-1]["close"],
        "volume": sum(row["volume"] for row in bucket),
    }
