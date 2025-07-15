import db
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from paths import HIGHLOW_CACHE_FILE

def is_market_open(session_start="09:14:58", session_end="15:30:05"):
    TZ = ZoneInfo("Asia/Kolkata")
    now = datetime.now(TZ).time()
    start = datetime.strptime(session_start, "%H:%M:%S").time()
    end = datetime.strptime(session_end, "%H:%M:%S").time()
    return start <= now <= end

def fetch_and_cache_highlow_levels(db_path=db.DB_PATH, cache_file=HIGHLOW_CACHE_FILE):
    """
    Fetch rolling weekly/monthly high/low for all symbols based on N trading sessions and cache to a daily JSON file.
    """
    try:
        weekly = db.get_high_low_for_last_n_sessions_all_symbols(7, db_path)
        monthly = db.get_high_low_for_last_n_sessions_all_symbols(30, db_path)
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        cache = {
            "weekly": weekly,
            "monthly": monthly,
            "date": today.isoformat()
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"[analytics] High/low cache successfully written to {cache_file}")
        return cache
    except Exception as e:
        print(f"[analytics][ERROR] Failed to build/write high/low cache: {e}")
        raise

def load_highlow_cache(cache_file=HIGHLOW_CACHE_FILE):
    """
    Load the cached weekly/monthly high/low levels for all symbols from file.
    Returns a dict with "weekly", "monthly", "date" keys or None if file missing/invalid.
    """
    if not os.path.isfile(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache
    except Exception:
        return None

def get_highlow_levels_for_symbol(symbol, cache):
    """
    Helper to retrieve high/low for a symbol from cache dict.
    Returns: (week_high, week_low, week_sessions), (month_high, month_low, month_sessions)
    If not available (i.e., None), returns (None, None, 0).
    """
    weekly = cache.get("weekly", {}).get(symbol, {"high": None, "low": None, "num_sessions": 0})
    monthly = cache.get("monthly", {}).get(symbol, {"high": None, "low": None, "num_sessions": 0})
    return (weekly.get("high"), weekly.get("low"), weekly.get("num_sessions")), (monthly.get("high"), monthly.get("low"), monthly.get("num_sessions"))

def ensure_today_highlow_cache(db_path=db.DB_PATH, cache_file=HIGHLOW_CACHE_FILE):
    """
    Ensure that the high/low cache file exists and is for today. If not, fetch and cache new values.
    Returns the loaded cache.
    """
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    cache = load_highlow_cache(cache_file)
    if not cache or cache.get("date") != today:
        print(f"[analytics] High/low cache missing or not for today ({today}). Rebuilding...")
        cache = fetch_and_cache_highlow_levels(db_path=db_path, cache_file=cache_file)
    return cache