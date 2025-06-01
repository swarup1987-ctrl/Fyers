from tkinter import *
from tkinter import ttk, messagebox
import resampler
import threading
import queue
import sqlite3
from project_paths import data_path
from deepseek_strategy import Intraday15MinStrategy
from tpm import TPMStrategy
from corb import CORBStrategy
from orb import ORBStrategy
from vrsi import VRSIStrategy
from vema import VEMAStrategy  # <-- Import the new strategy
from walkforward import WalkForwardBacktester
import numbers
from datetime import datetime

# Timezone support
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

def to_human_time(ts):
    try:
        if ts is None:
            return ""
        if isinstance(ts, numbers.Number):
            if ts != ts or ts == float('inf') or ts == float('-inf'):
                return ""
            if ts < 100000000:
                return str(ts)
            dt_utc = datetime.utcfromtimestamp(float(ts))
            try:
                dt_ist = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
            except Exception:
                import pytz
                dt_ist = pytz.UTC.localize(dt_utc).astimezone(IST)
            return dt_ist.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(ts, str) and ts.isdigit():
            dt_utc = datetime.utcfromtimestamp(float(ts))
            try:
                dt_ist = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
            except Exception:
                import pytz
                dt_ist = pytz.UTC.localize(dt_utc).astimezone(IST)
            return dt_ist.strftime('%Y-%m-%d %H:%M:%S')
        return str(ts)
    except Exception:
        return str(ts)

def open_backtest_window(parent):
    from gui_main import symbol_data  # Use symbol master from main GUI

    global active_strategy_instance, active_strategy_key
    win = Toplevel(parent)
    win.title("Backtest Interface")
    win.geometry("1200x1000")

    Label(win, text="Backtest interface").pack(pady=10)

    search_frame = Frame(win)
    search_frame.pack(pady=10)
    Label(search_frame, text="Search or Enter Stock Name/Symbol:").pack(side=LEFT, padx=5)
    search_entry = Entry(search_frame, width=30)
    search_entry.pack(side=LEFT, padx=5)

    result_list = Listbox(win, width=60, height=7)
    result_list.pack(pady=5)

    selected_symbol_var = StringVar()
    selected_exsym_var = StringVar()
    active_interval_var = StringVar(value="5min")

    def on_search(*_):
        query = search_entry.get().strip().lower()
        result_list.delete(0, END)
        for symbol, data in symbol_data.items():
            exsym = data.get("exSymName", "").lower()
            if query in symbol.lower() or query in exsym:
                exsym_disp = data.get("exSymName", "")
                result_list.insert(END, f"{symbol} - {exsym_disp}")

    def on_select(_):
        selection = result_list.curselection()
        if selection:
            selected = result_list.get(selection[0])
            symbol, exsym_disp = selected.split(" - ", 1)
            selected_symbol_var.set(symbol)
            selected_exsym_var.set(exsym_disp)
            success = resampler.fetch_and_store_symbol(symbol)
            if success:
                result_label.config(text=f"Fetched & resampled data for '{symbol}'.", fg="green")
            else:
                conn = sqlite3.connect(str(data_path("historical_data.db")))
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT symbol FROM historical_data LIMIT 5")
                sample = [row[0] for row in cur.fetchall()]
                conn.close()
                result_label.config(
                    text=f"Symbol '{symbol}' is NOT present in historical_data.db.\nSample symbols: {sample}", fg="red")

    search_entry.bind("<KeyRelease>", on_search)
    result_list.bind("<<ListboxSelect>>", on_select)

    Label(win, textvariable=selected_symbol_var, font=("Arial", 11), fg="blue").pack()
    Label(win, textvariable=selected_exsym_var, font=("Arial", 10), fg="gray").pack()

    result_label = Label(win, text="", fg="blue", font=("Arial", 10))
    result_label.pack(pady=10)

    timeframe_frame = Frame(win)
    timeframe_frame.pack(pady=10)
    Label(timeframe_frame, text="Select Resample Timeframe:").pack(side=LEFT, padx=5)
    timeframe_var = StringVar(value="5 min")
    timeframe_dropdown = ttk.Combobox(
        timeframe_frame,
        textvariable=timeframe_var,
        values=["5 min", "15 min", "1 hour"],
        state="readonly",
        width=10
    )
    timeframe_dropdown.pack(side=LEFT, padx=5)

    def on_interval_change(*_):
        val = timeframe_var.get()
        if val == "5 min":
            active_interval_var.set("5min")
        elif val == "15 min":
            active_interval_var.set("15min")
        elif val == "1 hour":
            active_interval_var.set("1hour")
        else:
            active_interval_var.set("5min")

    timeframe_dropdown.bind("<<ComboboxSelected>>", on_interval_change)

    strategy_frame = Frame(win)
    strategy_frame.pack(pady=10)
    Label(strategy_frame, text="Select Strategy:").pack(side=LEFT, padx=5)
    strategy_var = StringVar(value="deep.boll.vwap.rsi.macd")
    strategy_dropdown = ttk.Combobox(
        strategy_frame,
        textvariable=strategy_var,
        values=[
            "deep.boll.vwap.rsi.macd",
            "tpm.ema.rsi.vol",
            "corb.breakout",
            "orb.riskreward",
            "vrsi.vwap.rsi.pivot",
            "vema.ema.cross",  # <-- Add VEMA strategy to the list
        ],
        state="readonly",
        width=30
    )
    strategy_dropdown.pack(side=LEFT, padx=5)

    # VEMA parameter fields
    vema_frame = Frame(win)
    vema_frame.pack(pady=10)
    Label(vema_frame, text="VEMA Stop Loss %:").pack(side=LEFT, padx=5)
    vema_stop_loss_var = StringVar(value="0.5")
    Entry(vema_frame, width=5, textvariable=vema_stop_loss_var).pack(side=LEFT, padx=2)
    Label(vema_frame, text="VEMA Target %:").pack(side=LEFT, padx=5)
    vema_target_var = StringVar(value="0.8")
    Entry(vema_frame, width=5, textvariable=vema_target_var).pack(side=LEFT, padx=2)
    Label(vema_frame, text="VEMA Vol MA Window:").pack(side=LEFT, padx=5)
    vema_vol_window_var = StringVar(value="20")
    Entry(vema_frame, width=5, textvariable=vema_vol_window_var).pack(side=LEFT, padx=2)
    Label(vema_frame, text="VEMA Daily Loss Cap %:").pack(side=LEFT, padx=5)
    vema_daily_loss_var = StringVar(value="0.01")
    Entry(vema_frame, width=7, textvariable=vema_daily_loss_var).pack(side=LEFT, padx=2)
    Label(vema_frame, text="VEMA EMA Fast:").pack(side=LEFT, padx=5)
    vema_ema_fast_var = StringVar(value="9")
    Entry(vema_frame, width=5, textvariable=vema_ema_fast_var).pack(side=LEFT, padx=2)
    Label(vema_frame, text="VEMA EMA Slow:").pack(side=LEFT, padx=5)
    vema_ema_slow_var = StringVar(value="21")
    Entry(vema_frame, width=5, textvariable=vema_ema_slow_var).pack(side=LEFT, padx=2)

    opt_frame = Frame(win)
    opt_frame.pack(pady=10)
    opt_var = BooleanVar(value=True)
    opt_check = Checkbutton(opt_frame, text="Parameter Optimization (fit on training set)", variable=opt_var)
    opt_check.pack(side=LEFT, padx=5)

    Label(opt_frame, text="Train Window (days):").pack(side=LEFT, padx=5)
    train_days_var = IntVar(value=60)
    Entry(opt_frame, width=4, textvariable=train_days_var).pack(side=LEFT, padx=2)
    Label(opt_frame, text="Test Window (days):").pack(side=LEFT, padx=5)
    test_days_var = IntVar(value=10)
    Entry(opt_frame, width=4, textvariable=test_days_var).pack(side=LEFT, padx=2)
    Label(opt_frame, text="Step Size (days):").pack(side=LEFT, padx=5)
    step_days_var = IntVar(value=5)
    Entry(opt_frame, width=4, textvariable=step_days_var).pack(side=LEFT, padx=2)

    thresh_ckpt_frame = Frame(win)
    thresh_ckpt_frame.pack(pady=10)
    Label(thresh_ckpt_frame, text="Threshold (%):").pack(side=LEFT, padx=5)
    threshold_var = StringVar(value="")
    Entry(thresh_ckpt_frame, width=6, textvariable=threshold_var).pack(side=LEFT, padx=2)
    Label(thresh_ckpt_frame, text="Checkpoint (%):").pack(side=LEFT, padx=5)
    checkpoint_var = StringVar(value="")
    Entry(thresh_ckpt_frame, width=6, textvariable=checkpoint_var).pack(side=LEFT, padx=2)

    orb_rr_frame = Frame(win)
    orb_rr_frame.pack(pady=10)
    Label(orb_rr_frame, text="ORB RR Ratio (e.g. 2 for 1:2, 3 for 1:3):").pack(side=LEFT, padx=5)
    orb_rr_var = StringVar(value="2.0")
    Entry(orb_rr_frame, width=6, textvariable=orb_rr_var).pack(side=LEFT, padx=2)

    vrsi_frame = Frame(win)
    vrsi_frame.pack(pady=10)
    Label(vrsi_frame, text="VRSI RSI Period:").pack(side=LEFT, padx=5)
    vrsi_rsi_period_var = StringVar(value="14")
    Entry(vrsi_frame, width=5, textvariable=vrsi_rsi_period_var).pack(side=LEFT, padx=2)
    Label(vrsi_frame, text="VRSI Daily Loss Cap % (e.g. 0.01):").pack(side=LEFT, padx=5)
    vrsi_daily_loss_var = StringVar(value="0.01")
    Entry(vrsi_frame, width=7, textvariable=vrsi_daily_loss_var).pack(side=LEFT, padx=2)

    sampler_frame = Frame(win)
    sampler_frame.pack(pady=8)
    Label(sampler_frame, text="Parameter Search Sampler:").pack(side=LEFT, padx=5)
    sampler_var = StringVar(value="bayesian")
    sampler_dropdown = ttk.Combobox(
        sampler_frame,
        textvariable=sampler_var,
        values=["bayesian", "random", "grid"],
        state="readonly",
        width=12
    )
    sampler_dropdown.pack(side=LEFT, padx=5)

    Label(sampler_frame, text="n_trials:").pack(side=LEFT, padx=3)
    n_trials_var = IntVar(value=50)
    Entry(sampler_frame, width=5, textvariable=n_trials_var).pack(side=LEFT, padx=2)
    Label(sampler_frame, text="n_random_trials:").pack(side=LEFT, padx=3)
    n_random_trials_var = IntVar(value=20)
    Entry(sampler_frame, width=5, textvariable=n_random_trials_var).pack(side=LEFT, padx=2)

    grid_label = Label(opt_frame, text="(Using expanded parameter grid for optimization)")
    grid_label.pack(side=LEFT, padx=5)

    global active_strategy_instance, active_strategy_key
    active_strategy_instance = None
    active_strategy_key = None

    def on_strategy_change(*_):
        global active_strategy_instance, active_strategy_key
        selected = strategy_var.get()
        active_strategy_key = selected
        if selected == "deep.boll.vwap.rsi.macd":
            active_strategy_instance = Intraday15MinStrategy()
            result_label.config(text="Strategy 'deep.boll.vwap.rsi.macd' loaded and ready.", fg="green")
        elif selected == "tpm.ema.rsi.vol":
            active_strategy_instance = TPMStrategy()
            result_label.config(text="Strategy 'tpm.ema.rsi.vol' loaded and ready.", fg="green")
        elif selected == "corb.breakout":
            active_strategy_instance = CORBStrategy()
            result_label.config(text="Strategy 'corb.breakout' loaded and ready.", fg="green")
        elif selected == "orb.riskreward":
            try:
                rr_val = float(orb_rr_var.get())
            except Exception:
                rr_val = 2.0
            active_strategy_instance = ORBStrategy(rr_ratio=rr_val)
            result_label.config(text="Strategy 'orb.riskreward' loaded and ready.", fg="green")
        elif selected == "vrsi.vwap.rsi.pivot":
            try:
                rsi_period = int(vrsi_rsi_period_var.get())
            except Exception:
                rsi_period = 14
            try:
                daily_loss = float(vrsi_daily_loss_var.get())
            except Exception:
                daily_loss = 0.01
            active_strategy_instance = VRSIStrategy(rsi_period=rsi_period, daily_loss_pct=daily_loss)
            result_label.config(text="Strategy 'vrsi.vwap.rsi.pivot' loaded and ready.", fg="green")
        elif selected == "vema.ema.cross":
            try:
                stop_loss = float(vema_stop_loss_var.get())
            except Exception:
                stop_loss = 0.5
            try:
                target = float(vema_target_var.get())
            except Exception:
                target = 0.8
            try:
                vol_window = int(vema_vol_window_var.get())
            except Exception:
                vol_window = 20
            try:
                daily_loss = float(vema_daily_loss_var.get())
            except Exception:
                daily_loss = 0.01
            try:
                ema_fast = int(vema_ema_fast_var.get())
            except Exception:
                ema_fast = 9
            try:
                ema_slow = int(vema_ema_slow_var.get())
            except Exception:
                ema_slow = 21
            active_strategy_instance = VEMAStrategy(
                stop_loss_pct=stop_loss,
                target_pct=target,
                vol_window=vol_window,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                daily_loss_cap=daily_loss
            )
            result_label.config(text="Strategy 'vema.ema.cross' loaded and ready.", fg="green")
        else:
            active_strategy_instance = None
            result_label.config(text="No strategy loaded", fg="red")

    strategy_dropdown.bind("<<ComboboxSelected>>", on_strategy_change)
    result_label = Label(win, text="", fg="blue", font=("Arial", 10))
    result_label.pack(pady=10)
    on_strategy_change()

    def show_current_resampled_data():
        interval = active_interval_var.get()
        if interval == "5min":
            data = resampler.get_resampled_data("5min")
        elif interval == "15min":
            data = resampler.get_resampled_data("15min")
        elif interval == "1hour":
            data = resampler.get_resampled_data("1hour")
        else:
            data = []
        if not data:
            messagebox.showinfo("Info", "No data to display. Please select a symbol first.")
            return
        display_win = Toplevel(win)
        display_win.title(f"Resampled Data ({interval})")
        display_win.geometry("1100x600")

        columns = ["id", "symbol", "timestamp", "open", "high", "low", "close", "volume"]
        tree = ttk.Treeview(display_win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            if col == "symbol":
                tree.column(col, width=120)
            elif col == "timestamp":
                tree.column(col, width=140)
            else:
                tree.column(col, width=80)
        tree.pack(fill=BOTH, expand=True)
        max_rows = 1000
        for i, row in enumerate(data):
            if i >= max_rows:
                break
            ts = row["timestamp"]
            ts_str = to_human_time(ts)
            tree.insert("", END, values=(
                row.get("id", ""), row.get("symbol", ""), ts_str,
                row.get("open", ""), row.get("high", ""), row.get("low", ""),
                row.get("close", ""), row.get("volume", "")
            ))
        if len(data) > max_rows:
            Label(display_win, text=f"Showing first {max_rows} rows (out of {len(data)})", fg="red").pack()
        Button(display_win, text="Close", command=display_win.destroy).pack(pady=10)

    def on_backtest():
        symbol = selected_symbol_var.get()
        interval = active_interval_var.get()
        strategy_key = strategy_var.get()
        param_optimization = opt_var.get()
        train_days = train_days_var.get()
        test_days = test_days_var.get()
        step_days = step_days_var.get()
        sampler = sampler_var.get()
        n_trials = n_trials_var.get()
        n_random_trials = n_random_trials_var.get()
        threshold_str = threshold_var.get().strip()
        checkpoint_str = checkpoint_var.get().strip()
        orb_rr_str = orb_rr_var.get().strip()
        vrsi_rsi_period_str = vrsi_rsi_period_var.get().strip()
        vrsi_daily_loss_str = vrsi_daily_loss_var.get().strip()
        vema_stop_loss_str = vema_stop_loss_var.get().strip()
        vema_target_str = vema_target_var.get().strip()
        vema_vol_window_str = vema_vol_window_var.get().strip()
        vema_daily_loss_str = vema_daily_loss_var.get().strip()
        vema_ema_fast_str = vema_ema_fast_var.get().strip()
        vema_ema_slow_str = vema_ema_slow_var.get().strip()

        if not symbol:
            messagebox.showwarning("Warning", "Please select a symbol.")
            return
        if not interval:
            messagebox.showwarning("Warning", "Please select an interval.")
            return
        if not strategy_key:
            messagebox.showwarning("Warning", "Please select a strategy.")
            return

        use_trailing = bool(threshold_str and checkpoint_str)
        threshold_pct = None
        checkpoint_pct = None

        if use_trailing:
            try:
                threshold_pct = float(threshold_str)
                checkpoint_pct = float(checkpoint_str)
                if threshold_pct <= 0 or checkpoint_pct <= 0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("Warning", "Please enter valid positive numbers for Threshold and Checkpoint (%), or leave both empty.")
                return

        orb_rr_ratio = None
        if strategy_key == "orb.riskreward":
            try:
                orb_rr_ratio = float(orb_rr_str)
                if orb_rr_ratio <= 0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("Warning", "Please enter a valid RR ratio for ORB (e.g. 2 or 3).")
                return

        vrsi_rsi_period = None
        vrsi_daily_loss = None
        if strategy_key == "vrsi.vwap.rsi.pivot":
            try:
                vrsi_rsi_period = int(vrsi_rsi_period_str)
                if vrsi_rsi_period <= 0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("Warning", "Please enter a valid RSI period for VRSI.")
                return
            try:
                vrsi_daily_loss = float(vrsi_daily_loss_str)
                if vrsi_daily_loss < 0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("Warning", "Please enter a valid VRSI daily loss cap (e.g. 0.01).")
                return

        vema_stop_loss = None
        vema_target = None
        vema_vol_window = None
        vema_daily_loss = None
        vema_ema_fast = None
        vema_ema_slow = None
        if strategy_key == "vema.ema.cross":
            try:
                vema_stop_loss = float(vema_stop_loss_str)
                if vema_stop_loss <= 0:
                    raise ValueError
            except Exception:
                vema_stop_loss = 0.5
            try:
                vema_target = float(vema_target_str)
                if vema_target <= 0:
                    raise ValueError
            except Exception:
                vema_target = 0.8
            try:
                vema_vol_window = int(vema_vol_window_str)
                if vema_vol_window <= 0:
                    raise ValueError
            except Exception:
                vema_vol_window = 20
            try:
                vema_daily_loss = float(vema_daily_loss_str)
                if vema_daily_loss < 0:
                    raise ValueError
            except Exception:
                vema_daily_loss = 0.01
            try:
                vema_ema_fast = int(vema_ema_fast_str)
                if vema_ema_fast <= 0:
                    raise ValueError
            except Exception:
                vema_ema_fast = 9
            try:
                vema_ema_slow = int(vema_ema_slow_str)
                if vema_ema_slow <= 0:
                    raise ValueError
            except Exception:
                vema_ema_slow = 21

        progress_win = Toplevel(win)
        progress_win.title("Backtest Progress")
        progress_win.geometry("400x120")
        Label(progress_win, text="Running walk-forward backtest...").pack(pady=10)
        pb = ttk.Progressbar(progress_win, orient="horizontal", mode="determinate", length=320)
        pb.pack(pady=10)
        progress_label = Label(progress_win, text="Initializing ...")
        progress_label.pack()

        progress_q = queue.Queue()

        def update_progress():
            try:
                while not progress_q.empty():
                    val, msg = progress_q.get_nowait()
                    pb["value"] = val
                    pb.update_idletasks()
                    progress_label.config(text=msg)
                if pb["value"] < 100:
                    progress_win.after(250, update_progress)
                else:
                    progress_label.config(text="Walk-forward finished!")
            except Exception as e:
                progress_label.config(text=f"Error: {e}")

        def do_backtest():
            try:
                def progress_callback(window_num, total_windows, window_stat=None):
                    val = int(window_num / total_windows * 100)
                    msg = f"Window {window_num}/{total_windows}"
                    if window_stat and isinstance(window_stat, dict):
                        msg += f": trades={window_stat.get('num_trades','?')}, profit={window_stat.get('gross_profit', '?'):.2f}"
                    progress_q.put((val, msg))
                extra_kwargs = {}
                if strategy_key == "tpm.ema.rsi.vol":
                    extra_kwargs.update({
                        "profit_target_pct": threshold_pct if use_trailing else None,
                        "stop_loss_pct": checkpoint_pct if use_trailing else None,
                        "trailing_target_pct": threshold_pct if use_trailing else None,
                        "trailing_stop_pct": checkpoint_pct if use_trailing else None,
                    })
                elif strategy_key == "corb.breakout":
                    pass
                elif strategy_key == "orb.riskreward":
                    extra_kwargs.update({
                        "orb_rr_ratio": orb_rr_ratio
                    })
                elif strategy_key == "vrsi.vwap.rsi.pivot":
                    extra_kwargs.update({
                        "vrsi_rsi_period": vrsi_rsi_period,
                        "vrsi_daily_loss_pct": vrsi_daily_loss
                    })
                elif strategy_key == "vema.ema.cross":
                    extra_kwargs.update({
                        "vema_stop_loss_pct": vema_stop_loss,
                        "vema_target_pct": vema_target,
                        "vema_vol_window": vema_vol_window,
                        "vema_daily_loss_cap": vema_daily_loss,
                        "vema_ema_fast": vema_ema_fast,
                        "vema_ema_slow": vema_ema_slow,
                    })
                backtester = WalkForwardBacktester(
                    strategy_key=strategy_key,
                    interval=interval,
                    symbol=symbol,
                    capital=10000,
                    param_optimization=param_optimization,
                    train_days=train_days,
                    test_days=test_days,
                    step_days=step_days,
                    progress_callback=progress_callback,
                    param_sampler=sampler,
                    n_trials=n_trials,
                    n_random_trials=n_random_trials,
                    initial_threshold_pct=threshold_pct if strategy_key == "deep.boll.vwap.rsi.macd" else None,
                    initial_checkpoint_pct=checkpoint_pct if strategy_key == "deep.boll.vwap.rsi.macd" else None,
                    disable_trailing=not use_trailing if strategy_key in ["deep.boll.vwap.rsi.macd", "tpm.ema.rsi.vol"] else True,
                    **extra_kwargs
                )
                results = backtester.run_backtest()
                stats = results["stats"]
                trades = results["trades"]
                per_window = results.get("per_window", [])
                progress_q.put((100, "Done"))
                win.after(0, lambda: (progress_win.destroy(), show_backtest_results(win, stats, trades, per_window)))
            except Exception as e:
                win.after(0, lambda err=e: (progress_win.destroy(), messagebox.showerror("Error", f"Backtest failed: {err}")))

        pb["value"] = 0
        threading.Thread(target=do_backtest, daemon=True).start()
        win.after(250, update_progress)

    Button(win, text="Show", command=show_current_resampled_data).pack(pady=5)
    Button(win, text="Backtest", command=on_backtest).pack(pady=5)
    Button(win, text="Close", command=win.destroy).pack(pady=30)

def show_backtest_results(parent, stats, trades, per_window=None):
    win = Toplevel(parent)
    win.title("Backtest Results")
    win.geometry("1350x900")
    stat_text = ""
    for k, v in stats.items():
        if isinstance(v, float):
            stat_text += f"{k}: {v:.3f}\n"
        else:
            stat_text += f"{k}: {v}\n"

    trade_lines = []
    header = "Entry Time,Exit Time,Direction,P&L,Exit Reason"
    trade_lines.append(header)
    for t in trades:
        entry_time = to_human_time(t.get('entry_time', ''))
        exit_time = to_human_time(t.get('exit_time', ''))
        trade_lines.append(f"{entry_time},{exit_time},{t['direction']},{round(t['pl'],2)},{t['exit_reason']}")
    trades_text = "\n".join(trade_lines)

    results_box = Text(win, height=15, width=150, wrap='none', font=("Courier New", 10))
    results_box.insert("1.0", "Backtest Statistics:\n" + stat_text + "\nTrades:\n" + trades_text)
    results_box.config(state="disabled")
    results_box.pack(pady=6)

    def copy_results():
        win.clipboard_clear()
        win.clipboard_append(results_box.get("1.0", END).rstrip())
        win.update()

    Button(win, text="Copy Results", command=copy_results).pack(pady=3)

    if per_window and len(per_window):
        Label(win, text="Walk-Forward Window Details (including optimal params)", font=("Arial", 11, "bold")).pack(pady=10)
        pw_frame = Frame(win)
        pw_frame.pack(fill=BOTH, expand=True, padx=4, pady=4)
        columns = [
            "window", "train_start", "test_start", "num_trades", "win", "loss",
            "gross_profit", "gross_loss", "max_drawdown", "capital_end", "best_params", "train_stats"
        ]
        pw_tree = ttk.Treeview(pw_frame, columns=columns, show="headings", height=12)
        for col in columns:
            pw_tree.heading(col, text=col)
            if col in ("best_params", "train_stats"):
                pw_tree.column(col, width=400, minwidth=200, stretch=True)
            else:
                pw_tree.column(col, width=120, minwidth=80, stretch=True)
        for w in per_window:
            pw_tree.insert(
                "", END,
                values=[
                    w.get("window", ""),
                    to_human_time(w.get("train_start", "")),
                    to_human_time(w.get("test_start", "")),
                    w.get("num_trades", ""),
                    w.get("win", ""),
                    w.get("loss", ""),
                    round(w.get("gross_profit", 0), 2) if w.get("gross_profit") is not None else "",
                    round(w.get("gross_loss", 0), 2) if w.get("gross_loss") is not None else "",
                    round(w.get("max_drawdown", 0), 2) if w.get("max_drawdown") is not None else "",
                    round(w.get("capital_end", 0), 2) if w.get("capital_end") is not None else "",
                    w.get("best_params", ""),
                    str(w.get("train_stats", ""))
                ]
            )
        pw_tree.pack(fill=BOTH, expand=True, side=TOP)
        h_scroll = ttk.Scrollbar(pw_frame, orient="horizontal", command=pw_tree.xview)
        pw_tree.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(fill=X, side=BOTTOM)

    Button(win, text="Close", command=win.destroy).pack(pady=10)
