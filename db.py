import sqlite3
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

DB_PATH = "ticks_data.db"

TICK_FIELDS = [
    "symbol",
    "exch_feed_time",  # Exchange feed time as IST epoch (seconds since 1970-01-01 05:30:00 IST)
    "ltp",
    "vol_traded_today",
    "last_traded_time",
    "bid_size",
    "ask_size",
    "bid_price",
    "ask_price",
    "tot_buy_qty",
    "tot_sell_qty",
    "avg_trade_price",
    "lower_ckt",
    "upper_ckt",
    "received_time"
]

def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exch_feed_time INTEGER NOT NULL,
            ltp REAL,
            vol_traded_today INTEGER,
            last_traded_time INTEGER,
            bid_size INTEGER,
            ask_size INTEGER,
            bid_price REAL,
            ask_price REAL,
            tot_buy_qty INTEGER,
            tot_sell_qty INTEGER,
            avg_trade_price REAL,
            lower_ckt REAL,
            upper_ckt REAL,
            received_time INTEGER,
            UNIQUE(symbol, exch_feed_time)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_time ON ticks(symbol, exch_feed_time);
    """)
    conn.commit()
    conn.close()

def insert_tick(tick: Dict[str, Any], conn: Optional[sqlite3.Connection] = None, db_path: str = DB_PATH):
    fields = TICK_FIELDS
    values = [tick.get(field) for field in fields]
    placeholders = ", ".join(["?"] * len(fields))
    sql = f"INSERT OR IGNORE INTO ticks ({', '.join(fields)}) VALUES ({placeholders})"
    if conn is not None:
        conn.execute(sql, values)
        conn.commit()
    else:
        with sqlite3.connect(db_path) as _conn:
            _conn.execute(sql, values)
            _conn.commit()

def insert_ticks_batch(ticks: List[Dict[str, Any]], conn: Optional[sqlite3.Connection] = None, db_path: str = DB_PATH):
    if not ticks:
        return
    fields = TICK_FIELDS
    placeholders = ", ".join(["?"] * len(fields))
    sql = f"INSERT OR IGNORE INTO ticks ({', '.join(fields)}) VALUES ({placeholders})"
    values = [[tick.get(field) for field in fields] for tick in ticks]
    if conn is not None:
        conn.executemany(sql, values)
        conn.commit()
    else:
        with sqlite3.connect(db_path) as _conn:
            _conn.executemany(sql, values)
            _conn.commit()

class TickDBWorker(threading.Thread):
    def __init__(self, db_path=DB_PATH, batch_size=1):
        super().__init__(daemon=True)
        import queue
        self.db_path = db_path
        self.queue = queue.Queue()
        self._stop_signal = object()
        self.batch_size = batch_size

    def run(self):
        conn = sqlite3.connect(self.db_path)
        buffer = []
        while True:
            try:
                item = self.queue.get(timeout=0.2)
                if item is self._stop_signal:
                    if buffer:
                        insert_ticks_batch(buffer, conn=conn)
                    break
                buffer.append(item)
                if len(buffer) >= self.batch_size:
                    insert_ticks_batch(buffer, conn=conn)
                    buffer.clear()
            except Exception:
                if buffer:
                    insert_ticks_batch(buffer, conn=conn)
                    buffer.clear()
        conn.close()

    def put(self, tick):
        self.queue.put(tick)

    def stop(self):
        self.queue.put(self._stop_signal)
        self.join()

def get_latest_ticks(db_path: str = DB_PATH) -> Dict[str, dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t1.*
        FROM ticks t1
        INNER JOIN (
            SELECT symbol, MAX(exch_feed_time) AS max_time
            FROM ticks
            GROUP BY symbol
        ) t2
        ON t1.symbol = t2.symbol AND t1.exch_feed_time = t2.max_time
    """)
    results = cursor.fetchall()
    conn.close()
    ticks = {}
    for row in results:
        tick = {field: row[field] for field in TICK_FIELDS}
        ticks[tick["symbol"]] = tick
    return ticks

def get_all_symbols(db_path: str = DB_PATH) -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ticks")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def get_high_low_for_last_n_sessions(symbol: str, n: int, db_path: str = DB_PATH) -> Tuple[Optional[float], Optional[float], int]:
    """
    Return (high, low, num_sessions_with_data) for symbol in the last `n` trading sessions (dates with data).
    If no data, returns (None, None, 0)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get the last n unique trading dates with data for this symbol (most recent first)
    cursor.execute(
        """
        SELECT DISTINCT date((exch_feed_time + 19800), 'unixepoch') AS trading_date
        FROM ticks
        WHERE symbol=?
        ORDER BY trading_date DESC
        LIMIT ?
        """,
        (symbol, n)
    )
    dates = [row[0] for row in cursor.fetchall()]
    if not dates:
        conn.close()
        return (None, None, 0)
    # Query max/min ltp for these sessions
    placeholders = ",".join(["?"] * len(dates))
    query = f"""
        SELECT MAX(ltp), MIN(ltp), COUNT(DISTINCT date((exch_feed_time + 19800), 'unixepoch'))
        FROM ticks
        WHERE symbol=? AND date((exch_feed_time + 19800), 'unixepoch') IN ({placeholders})
    """
    cursor.execute(query, (symbol, *dates))
    row = cursor.fetchone()
    conn.close()
    if not row or (row[0] is None or row[1] is None):
        return (None, None, 0)
    return (row[0], row[1], row[2])

def get_high_low_for_last_n_sessions_all_symbols(n: int, db_path: str = DB_PATH) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Return a dict: {symbol: {"high": ..., "low": ..., "num_sessions": ...}} for all symbols in the last n trading sessions.
    If no data for a symbol, "high" and "low" will be None.
    """
    symbols = get_all_symbols(db_path)
    output = {}
    for symbol in symbols:
        high, low, num_sessions = get_high_low_for_last_n_sessions(symbol, n, db_path)
        output[symbol] = {"high": high, "low": low, "num_sessions": num_sessions}
    return output