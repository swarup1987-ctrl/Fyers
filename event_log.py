import os
import csv
from datetime import datetime
from pathlib import Path

from paths import BASE_DIR

EVENT_LOG_DIR = os.path.join(BASE_DIR, "event_logs")
os.makedirs(EVENT_LOG_DIR, exist_ok=True)

def get_today_logfile():
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(EVENT_LOG_DIR, f"events_{today}.csv")

def log_event(event: dict):
    """Append an event dict to today's log file. Header written automatically if file is new."""
    logfile = get_today_logfile()
    file_exists = os.path.isfile(logfile)
    with open(logfile, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "symbol", "event_type", "ltp", "high", "low", "period"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(event)

def read_today_events():
    """Read and return all events for today as a list of dicts."""
    logfile = get_today_logfile()
    events = []
    if os.path.isfile(logfile):
        with open(logfile, mode="r", newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
    return events

def ensure_today_event_logfile():
    """
    Ensure that today's event log file exists (with header), even if empty.
    """
    logfile = get_today_logfile()
    if not os.path.isfile(logfile):
        with open(logfile, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "symbol", "event_type", "ltp", "high", "low", "period"
            ])
            writer.writeheader()