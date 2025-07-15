import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fyers_apiv3.FyersWebsocket import data_ws
from db import init_db, TickDBWorker
from config import config

# --- Config ---
SYMBOLS_FILE = "symbols.txt"
DB_PATH = "ticks_data.db"
TZ = ZoneInfo("Asia/Kolkata")
RECEIVED_TIME_FIELD = "received_time"  # Now saving as IST epoch

# --- Load .env and token ---
load_dotenv()

def load_symbols(path=SYMBOLS_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please create it with one symbol per line.")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def get_ist_epoch(dt_utc):
    """
    Convert a UTC datetime to IST epoch seconds (since 1970-01-01 05:30:00 IST).
    """
    ist_epoch = datetime(1970, 1, 1, 5, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    dt_ist = dt_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    return int((dt_ist - ist_epoch).total_seconds())

# --- WebSocket Collector Class ---
class TickCollector:
    def __init__(self, symbols, ws_access_token, db_worker):
        self.symbols = symbols
        self.ws_access_token = ws_access_token
        self.connected = threading.Event()
        self.stopped = threading.Event()
        self.tick_count = 0
        self.db_worker = db_worker

    def onopen(self):
        print("[WebSocket Open] Subscribing to symbols...")
        fyers.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
        self.connected.set()

    def onmessage(self, message):
        print("[WebSocket Message]", message)  # Streaming print remains
        try:
            if not isinstance(message, dict):
                print("Received non-dict message, skipping:", message)
                return
            tick = {k: message.get(k) for k in [
                "symbol", "exch_feed_time", "ltp", "vol_traded_today", "last_traded_time",
                "bid_size", "ask_size", "bid_price", "ask_price", "tot_buy_qty",
                "tot_sell_qty", "avg_trade_price", "lower_ckt", "upper_ckt"
            ]}
            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            tick[RECEIVED_TIME_FIELD] = get_ist_epoch(now_utc)
            # Send tick to DB worker (queue) for fast, non-blocking insert
            self.db_worker.put(tick)
            self.tick_count += 1
        except Exception as e:
            print("[Tick Insert Error]", e)

    def onerror(self, message):
        print("[WebSocket Error]", message)

    def onclose(self, message):
        print("[WebSocket Closed]", message)
        self.stopped.set()

    def run(self):
        global fyers
        print("Starting TickCollector with", len(self.symbols), "symbols")
        fyers = data_ws.FyersDataSocket(
            access_token=self.ws_access_token,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.onopen,
            on_close=self.onclose,
            on_error=self.onerror,
            on_message=self.onmessage,
            reconnect_retry=10
        )
        fyers.connect()
        self.stopped.wait()
        fyers.disconnect()

    def stop(self):
        print("[TickCollector] Stop signal received.")
        self.stopped.set()

# --- Main entrypoint ---
def main(return_collector=False):
    print("=== ws_collector.py started ===")
    config.ensure_tokens_loaded()
    access_token, _ = config.get_tokens()
    CLIENT_ID = config.CLIENT_ID
    print("Loaded CLIENT_ID:", CLIENT_ID)
    print("Loaded access_token:", access_token)
    if not CLIENT_ID or not access_token:
        print("Missing CLIENT_ID or access_token. Aborting.")
        return

    ws_access_token = f"{CLIENT_ID}:{access_token}"

    # 2. Load symbols
    try:
        symbols = load_symbols()
        print(f"Loaded {len(symbols)} symbols.")
    except Exception as e:
        print(e)
        return
    if not symbols:
        print("No symbols found in symbols.txt. Aborting.")
        return

    # 3. Init DB (if not already)
    init_db(DB_PATH)
    print("Database initialized.")

    # 4. Start DB worker thread
    db_worker = TickDBWorker(db_path=DB_PATH, batch_size=5)
    db_worker.start()
    print("DB worker thread started.")

    # 5. Start WebSocket collector immediately
    collector = TickCollector(symbols, ws_access_token, db_worker)
    ws_thread = threading.Thread(target=collector.run, daemon=True)
    ws_thread.start()

    if return_collector:
        # Return collector, db_worker, and ws_thread to controller for full management
        return collector, db_worker, ws_thread

    ws_thread.join(timeout=None)
    print(f"WebSocket closed. Total ticks received: {collector.tick_count}")

    # 6. Stop DB worker thread gracefully
    db_worker.stop()
    print("DB worker thread stopped.")

if __name__ == "__main__":
    main()