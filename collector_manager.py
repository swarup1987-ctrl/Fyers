import threading
import ws_collector
from datetime import datetime
from zoneinfo import ZoneInfo

class CollectorManager:
    def __init__(self, end_time_str="15:30:05"):
        self.thread = None
        self.collector_instance = None
        self.db_worker = None
        self.ws_thread = None
        self.end_time_str = end_time_str
        self.closed_callback = None  # Optional callback for UI message

    def _is_past_end_time(self):
        TZ = ZoneInfo("Asia/Kolkata")
        now = datetime.now(TZ).time()
        end_time = datetime.strptime(self.end_time_str, "%H:%M:%S").time()
        return now >= end_time

    def set_closed_callback(self, cb):
        """Register a callback to be called when session is over and start is refused."""
        self.closed_callback = cb

    def start(self):
        if self._is_past_end_time():
            msg = f"[CollectorManager] Refusing to start: current time is past end time ({self.end_time_str} IST)."
            print(msg)
            if self.closed_callback:
                self.closed_callback()
            return

        if self.thread and self.thread.is_alive():
            print("[CollectorManager] Collector is already running.")
            return

        def run_collector():
            result = ws_collector.main(return_collector=True)
            if result is not None:
                self.collector_instance, self.db_worker, self.ws_thread = result
                self.ws_thread.join()
                print("[CollectorManager] Collector thread exiting.")
            else:
                print("[CollectorManager] Collector failed to start.")

        self.thread = threading.Thread(target=run_collector, daemon=True)
        self.thread.start()
        print("[CollectorManager] Collector started.")

    def stop(self):
        if self.collector_instance:
            self.collector_instance.stop()
            print("[CollectorManager] Collector stop signal sent.")
        if self.ws_thread:
            self.ws_thread.join(timeout=10)
        if self.db_worker:
            self.db_worker.stop()
            print("[CollectorManager] DB worker stop signal sent.")

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()