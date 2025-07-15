import threading
import time
from typing import Dict, Any, Callable
from datetime import datetime
import csv
import os
from zoneinfo import ZoneInfo
import db
import analytics
import event_log
from paths import HIGHLOW_CACHE_FILE

class Screener(threading.Thread):
    CIRCUIT_RELOAD_INTERVAL = 600  # seconds

    WEEK_MIN_DAYS = 7
    MONTH_MIN_DAYS = 30

    SESSION_START = "09:14:58"
    SESSION_END = "15:30:05"

    PROXIMITY_THRESHOLD_PERCENT = 0.5  # Permanent internal threshold

    def __init__(
        self,
        db_path,
        notice_callback: Callable[[dict], None],
        circuit_file="daily_circuits.csv",
        session_start=SESSION_START,
        session_end=SESSION_END,
        highlow_cache_file=HIGHLOW_CACHE_FILE
    ):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.notice_callback = notice_callback
        self.threshold = self.PROXIMITY_THRESHOLD_PERCENT
        self.poll_interval = 2.0
        self.prev_ltp: Dict[str, float] = {}
        self.circuit_file = circuit_file
        self.circuits = self.load_circuit_file(circuit_file)
        self._last_circuit_reload = time.time()
        self._last_circuit_mtime = self.get_circuit_file_mtime()
        self.session_start = session_start
        self.session_end = session_end
        self._last_alert_reset_date = None

        # High/low event state per symbol
        self.highlow_event_state: Dict[str, Dict[str, Any]] = {}

        # Load highlow cache at init (ensure it's for today)
        self.highlow_cache_file = highlow_cache_file
        self.highlow_cache = analytics.ensure_today_highlow_cache(self.db_path, self.highlow_cache_file)

    def is_market_open(self):
        TZ = ZoneInfo("Asia/Kolkata")
        now = datetime.now(TZ).time()
        start = datetime.strptime(self.session_start, "%H:%M:%S").time()
        end = datetime.strptime(self.session_end, "%H:%M:%S").time()
        return start <= now <= end

    def get_circuit_file_mtime(self):
        try:
            return os.path.getmtime(self.circuit_file)
        except Exception:
            return None

    def load_circuit_file(self, path) -> Dict[str, Dict[str, float]]:
        circuits = {}
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row["symbol"].strip()
                    try:
                        upper = float(row["upper_ckt"].replace(",", ""))
                        lower = float(row["lower_ckt"].replace(",", ""))
                        circuits[symbol] = {"upper": upper, "lower": lower}
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Screener] Failed to load circuit file: {e}")
        return circuits

    def maybe_reload_circuits(self):
        now = time.time()
        if now - self._last_circuit_reload >= self.CIRCUIT_RELOAD_INTERVAL:
            current_mtime = self.get_circuit_file_mtime()
            if current_mtime != self._last_circuit_mtime:
                print("[Screener] Detected change in daily_circuits.csv, reloading circuit bands.")
                new_circuits = self.load_circuit_file(self.circuit_file)
                if new_circuits:
                    self.circuits = new_circuits
                    self._last_circuit_mtime = current_mtime
            self._last_circuit_reload = now

    def reset_all_alert_flags(self):
        # Called at the start of each new trading session (date)
        self.highlow_event_state.clear()  # Reset all tracking state

    def run(self):
        self._last_alert_reset_date = None
        # Ensure highlow cache is always for today (in case of long-running process or date change)
        if not self.highlow_cache or self.highlow_cache.get("date") != datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat():
            self.highlow_cache = analytics.ensure_today_highlow_cache(self.db_path, self.highlow_cache_file)

        while True:
            # Reset alert flags at the start of every new date (trading session)
            tz = ZoneInfo("Asia/Kolkata")
            today = datetime.now(tz).date()
            if today != self._last_alert_reset_date:
                self.reset_all_alert_flags()
                self._last_alert_reset_date = today

            if not self.is_market_open():
                time.sleep(30)
                continue
            try:
                self.maybe_reload_circuits()
                # Ensure cache is always for today if date changed
                if not self.highlow_cache or self.highlow_cache.get("date") != today.isoformat():
                    self.highlow_cache = analytics.ensure_today_highlow_cache(self.db_path, self.highlow_cache_file)
                latest_ticks = db.get_latest_ticks(self.db_path)
                for symbol, tick in latest_ticks.items():
                    ltp = tick.get("ltp")
                    circuit = self.circuits.get(symbol)
                    if ltp is not None and circuit:
                        upper_ckt = circuit["upper"]
                        lower_ckt = circuit["lower"]
                        if upper_ckt != 0 and lower_ckt != 0:
                            prev_ltp = self.prev_ltp.get(symbol, ltp)
                            event_type = self.detect_event(symbol, ltp, prev_ltp, upper_ckt, lower_ckt)
                            if event_type:
                                event = self._make_event_dict(symbol, event_type, ltp, upper_ckt, lower_ckt, "circuit")
                                event_log.log_event(event)
                                self.notice_callback(event)
                            self.prev_ltp[symbol] = ltp
                    self.check_highlow_alert(symbol, ltp)
            except Exception as e:
                print("[Screener Error]", e)
            time.sleep(self.poll_interval)

    def detect_event(self, symbol, ltp, prev_ltp, upper_ckt, lower_ckt):
        proximity = self.threshold / 100.0
        upper_threshold = upper_ckt * (1 - proximity)
        lower_threshold = lower_ckt * (1 + proximity)
        if upper_threshold <= ltp < upper_ckt:
            return "VERY CLOSE TO UPPER CIRCUIT"
        if lower_ckt < ltp <= lower_threshold:
            return "VERY CLOSE TO LOWER CIRCUIT"
        if prev_ltp < upper_ckt and ltp >= upper_ckt:
            return "CROSSING UPPER CIRCUIT"
        if prev_ltp > lower_ckt and ltp <= lower_ckt:
            return "CROSSING LOWER CIRCUIT"
        if ltp >= upper_ckt:
            return "CROSSED UPPER CIRCUIT"
        if ltp <= lower_ckt:
            return "CROSSED LOWER CIRCUIT"
        return None

    def check_highlow_alert(self, symbol, ltp):
        if ltp is None or not self.highlow_cache:
            return
        # State for this symbol
        state = self.highlow_event_state.setdefault(symbol, {
            "WEEKLY_HIGH_NEAR": False,
            "WEEKLY_HIGH_CROSSED": False,
            "WEEKLY_LOW_NEAR": False,
            "WEEKLY_LOW_CROSSED": False,
            "MONTHLY_HIGH_NEAR": False,
            "MONTHLY_HIGH_CROSSED": False,
            "MONTHLY_LOW_NEAR": False,
            "MONTHLY_LOW_CROSSED": False,
            # Track if price is in proximity threshold right now
            "IN_WEEKLY_HIGH_THRESHOLD": False,
            "IN_WEEKLY_LOW_THRESHOLD": False,
            "IN_MONTHLY_HIGH_THRESHOLD": False,
            "IN_MONTHLY_LOW_THRESHOLD": False,
        })
        (week_high, week_low, week_sessions), (month_high, month_low, month_sessions) = analytics.get_highlow_levels_for_symbol(symbol, self.highlow_cache)
        proximity = self.threshold / 100.0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Helper for event firing logic per level
        def handle_level(period, high, low, min_days, sessions):
            if high is None or low is None or sessions < min_days:
                return

            # Use correct flag prefixes: WEEKLY_ or MONTHLY_
            prefix = "WEEKLY" if period == "week" else "MONTHLY"

            # --- High Logic ---
            high_near_flag = f"{prefix}_HIGH_NEAR"
            high_crossed_flag = f"{prefix}_HIGH_CROSSED"
            in_high_threshold_flag = f"IN_{prefix}_HIGH_THRESHOLD"
            crossed_high_flag = state[high_crossed_flag]
            was_in_threshold = state[in_high_threshold_flag]

            high_threshold_lower = high * (1 - proximity)
            high_threshold_upper = high  # Do not include above high

            # Only consider for imminent if not already crossed in this session
            if not crossed_high_flag:
                # If price moves into [high_threshold_lower, high) from below
                if not was_in_threshold and high_threshold_lower <= ltp < high:
                    # Fire "very close" event
                    event = {
                        "timestamp": now,
                        "symbol": symbol,
                        "event_type": f"VERY CLOSE TO {prefix} HIGH",
                        "ltp": ltp,
                        "high": high,
                        "low": low,
                        "period": period
                    }
                    event_log.log_event(event)
                    self.notice_callback(event)
                    state[high_near_flag] = True
                state[in_high_threshold_flag] = high_threshold_lower <= ltp < high
            else:
                # After crossed, never fire imminent again this session
                state[in_high_threshold_flag] = False

            # Crossed event, only fire once per session
            if not crossed_high_flag and ltp >= high:
                event = {
                    "timestamp": now,
                    "symbol": symbol,
                    "event_type": f"CROSSED {prefix} HIGH",
                    "ltp": ltp,
                    "high": high,
                    "low": low,
                    "period": period
                }
                event_log.log_event(event)
                self.notice_callback(event)
                state[high_crossed_flag] = True
                state[high_near_flag] = False  # Once crossed, don't fire imminent again

            # If price leaves the threshold (either below or above), reset the "near" flag
            if not (high_threshold_lower <= ltp < high):
                state[high_near_flag] = False

            # --- Low Logic ---
            low_near_flag = f"{prefix}_LOW_NEAR"
            low_crossed_flag = f"{prefix}_LOW_CROSSED"
            in_low_threshold_flag = f"IN_{prefix}_LOW_THRESHOLD"
            crossed_low_flag = state[low_crossed_flag]
            was_in_low_threshold = state[in_low_threshold_flag]

            low_threshold_lower = low  # Do not include below low
            low_threshold_upper = low * (1 + proximity)

            if not crossed_low_flag:
                # If price moves into (low, low_threshold_upper] from above
                if not was_in_low_threshold and low < ltp <= low_threshold_upper:
                    event = {
                        "timestamp": now,
                        "symbol": symbol,
                        "event_type": f"VERY CLOSE TO {prefix} LOW",
                        "ltp": ltp,
                        "high": high,
                        "low": low,
                        "period": period
                    }
                    event_log.log_event(event)
                    self.notice_callback(event)
                    state[low_near_flag] = True
                state[in_low_threshold_flag] = low < ltp <= low_threshold_upper
            else:
                # After crossed, never fire imminent again this session
                state[in_low_threshold_flag] = False

            # Crossed event, only fire once per session
            if not crossed_low_flag and ltp <= low:
                event = {
                    "timestamp": now,
                    "symbol": symbol,
                    "event_type": f"CROSSED {prefix} LOW",
                    "ltp": ltp,
                    "high": high,
                    "low": low,
                    "period": period
                }
                event_log.log_event(event)
                self.notice_callback(event)
                state[low_crossed_flag] = True
                state[low_near_flag] = False  # Once crossed, don't fire imminent again

            # If price leaves the threshold (either above or below), reset the "near" flag
            if not (low < ltp <= low_threshold_upper):
                state[low_near_flag] = False

        handle_level("week", week_high, week_low, self.WEEK_MIN_DAYS, week_sessions)
        handle_level("month", month_high, month_low, self.MONTH_MIN_DAYS, month_sessions)

    def _make_event_dict(self, symbol, event_type, ltp, high, low, period):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "timestamp": now,
            "symbol": symbol,
            "event_type": event_type,
            "ltp": ltp,
            "high": high,
            "low": low,
            "period": period
        }