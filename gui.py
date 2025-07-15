from tkinter import *
from tkinter import ttk, messagebox
from fyers_service import FyersService
from collector_manager import CollectorManager
from auth import authenticate, generate_token
import local_auth_server
import ws_scheduler
from screener import Screener
import event_log

from datetime import datetime, time
from zoneinfo import ZoneInfo
import analytics
from paths import HIGHLOW_CACHE_FILE
from db import DB_PATH, init_db  # <-- import init_db

auth_http_server = None
auth_http_thread = None
fyers_service = FyersService()
collector_manager = CollectorManager(end_time_str="15:30:05")

def is_market_open(session_start="09:14:58", session_end="15:30:05"):
    TZ = ZoneInfo("Asia/Kolkata")
    now = datetime.now(TZ).time()
    start = datetime.strptime(session_start, "%H:%M:%S").time()
    end = datetime.strptime(session_end, "%H:%M:%S").time()
    return start <= now <= end

class NoticeBoard(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Event Noticeboard - Today's Events")
        self.geometry("900x400")
        self.resizable(True, True)

        columns = ("timestamp", "symbol", "event_type", "ltp", "high", "low", "period")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col.replace("_", " ").capitalize())
            self.tree.column(col, anchor="center", width=120)
        self.tree.pack(fill=BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=RIGHT, fill=Y)

        # Load today's events at startup
        self.load_events()

    def load_events(self):
        self.tree.delete(*self.tree.get_children())
        events = event_log.read_today_events()
        for event in events:
            row = (
                event.get("timestamp"),
                event.get("symbol"),
                event.get("event_type"),
                event.get("ltp"),
                event.get("high"),
                event.get("low"),
                event.get("period"),
            )
            self.tree.insert("", END, values=row)

    def add_event(self, event):
        row = (
            event.get("timestamp"),
            event.get("symbol"),
            event.get("event_type"),
            event.get("ltp"),
            event.get("high"),
            event.get("low"),
            event.get("period"),
        )
        self.tree.insert("", END, values=row)
        self.tree.yview_moveto(1.0)

def show_profile(parent):
    try:
        profile = fyers_service.get_profile()
    except Exception as e:
        show_error(parent, f"Error fetching profile: {str(e)}")
        return

    win = Toplevel(parent)
    win.title("Profile Details")
    win.geometry("600x400")
    Button(win, text="Close", command=win.destroy).pack()
    Label(win, text=str(profile), wraplength=550, justify="left").pack(pady=20)

def show_funds(parent):
    try:
        funds = fyers_service.get_funds()
    except Exception as e:
        show_error(parent, f"Error fetching funds: {str(e)}")
        return

    win = Toplevel(parent)
    win.title("Funds")
    win.geometry("600x400")
    Button(win, text="Close", command=win.destroy).pack()
    Label(win, text=str(funds), wraplength=550, justify="left").pack(pady=20)

def show_error(parent, message):
    win = Toplevel(parent)
    win.title("Error")
    win.geometry("400x200")
    Label(win, text=message, fg="red").pack(pady=20)
    Button(win, text="Close", command=win.destroy).pack()

def show_session_over_message():
    messagebox.showinfo(
        "Market Session Closed",
        "It is past the scheduled market closing time.\n\n"
        "Tick data collection will not start.\n"
        "Please try again during the next session window."
    )

def launch_gui():
    global auth_http_server, auth_http_thread

    main = Tk()
    main.title("Fyers Authentication")
    main.geometry("600x400")

    top = Frame(main)
    top.pack(fill=X, pady=5)
    Button(top, text="Profile", command=lambda: show_profile(main)).pack(side=LEFT, padx=10)
    Button(top, text="Funds", command=lambda: show_funds(main)).pack(side=LEFT, padx=10)

    Label(main, text="Enter Auth Code:").pack(pady=5)
    auth_entry = Entry(main, width=40)
    auth_entry.pack(pady=5)

    def set_auth_code_in_entry(auth_code):
        auth_entry.delete(0, END)
        auth_entry.insert(0, auth_code)
        messagebox.showinfo("Auth Code Received", "Authentication code was inserted automatically.")
        global auth_http_server, auth_http_thread
        if auth_http_server is not None and auth_http_thread is not None:
            local_auth_server.stop_auth_server(auth_http_server, auth_http_thread)
            auth_http_server = None
            auth_http_thread = None

    def on_authenticate():
        global auth_http_server, auth_http_thread
        if auth_http_server is None and auth_http_thread is None:
            auth_http_server, auth_http_thread = local_auth_server.start_auth_server(
                8000,
                lambda code: main.after(0, set_auth_code_in_entry, code)
            )
        authenticate()

    Button(main, text="Authenticate", command=on_authenticate).pack(pady=5)
    Button(main, text="Generate Access Token", command=lambda: generate_token(auth_entry.get())).pack(pady=5)

    collector_manager.set_closed_callback(lambda: main.after(0, show_session_over_message))

    # IMPORTANT: Ensure DB is initialized before any analytics cache operation!
    init_db(DB_PATH)

    # Ensure highlow cache and event log file are ready at every startup!
    analytics.ensure_today_highlow_cache(db_path=DB_PATH, cache_file=HIGHLOW_CACHE_FILE)
    event_log.ensure_today_event_logfile()

    ws_scheduler.run_ws_collector_at_schedule(
        start_time_str="09:14:58",
        end_time_str="15:30:05",
        start_callback=collector_manager.start,
        stop_callback=collector_manager.stop,
        db_path=DB_PATH  # <-- ensure pre-session cache build works!
    )

    # --- Noticeboard setup
    noticeboard = NoticeBoard(main)

    def notice_callback(event):
        noticeboard.add_event(event)

    # Session guard for live screener
    if is_market_open():
        screener = Screener(
            db_path=DB_PATH,
            notice_callback=notice_callback,
            highlow_cache_file=HIGHLOW_CACHE_FILE
        )
        screener.start()
    else:
        def show_guard_msg():
            messagebox.showinfo("Market Session Inactive", "Live screening and alerts are available only during market hours (09:15–15:30 IST).")
        main.after(1000, show_guard_msg)

    # Button to open/reload noticeboard window
    Button(main, text="Show Noticeboard", command=noticeboard.load_events).pack(pady=8)

    main.mainloop()

if __name__ == "__main__":
    launch_gui()