import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Centralized resource paths (relative to BASE_DIR)
TOKEN_FILE_PATH = os.path.join(BASE_DIR, "tokens.txt")
LOG_FILE_PATH = os.path.join(BASE_DIR, "fyers_auth_log.txt")
EVENT_LOG_DIR = os.path.join(BASE_DIR, "event_logs")
# High/Low cache file for each day (overwritten daily)
HIGHLOW_CACHE_FILE = os.path.join(BASE_DIR, "highlow_cache.json")