import numpy as np
import pandas as pd
import hashlib
import pickle
import os

class ParamCache:
    """
    Caches best parameters for a given hash of training data.
    """
    def __init__(self, fname="vrsi_param_cache.pkl"):
        self.fname = fname
        if os.path.exists(self.fname):
            try:
                with open(self.fname, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def get(self, data_hash):
        return self.cache.get(data_hash, None)

    def set(self, data_hash, params):
        self.cache[data_hash] = params
        with open(self.fname, "wb") as f:
            pickle.dump(self.cache, f)

def calc_vwap(df):
    q = df['close'] * df['volume']
    vwap = q.cumsum() / df['volume'].cumsum()
    return vwap

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_pivots(df):
    # Use previous day's OHLC to compute pivots for each day
    pivots = []
    prev_date = None
    prev_ohlc = None
    for idx, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"], unit="s")
        date = ts.date()
        if prev_date is not None and date != prev_date:
            # Calculate pivots from prev_ohlc
            high, low, close = prev_ohlc["high"], prev_ohlc["low"], prev_ohlc["close"]
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            pivots.append({
                "date": date,
                "pivot": pivot,
                "r1": r1,
                "s1": s1,
                "r2": r2,
                "s2": s2
            })
        prev_date = date
        prev_ohlc = row
    # Fill last day with NaNs (no pivots since no next day)
    pivots.append({
        "date": date,
        "pivot": np.nan, "r1": np.nan, "s1": np.nan, "r2": np.nan, "s2": np.nan
    })
    pivots_df = pd.DataFrame(pivots)
    # Remove duplicate dates (keep first)
    pivots_df = pivots_df[~pivots_df["date"].duplicated(keep="first")]
    pivots_df = pivots_df.set_index("date")
    return pivots_df

class VRSIStrategy:
    def __init__(self, rsi_period=14, vwap_window=None, stop_type="pivot", target_type="pivot", daily_loss_pct=0.01):
        self.rsi_period = rsi_period
        self.vwap_window = vwap_window
        self.stop_type = stop_type
        self.target_type = target_type
        self.daily_loss_pct = daily_loss_pct
        self._params = {
            "rsi_period": rsi_period,
            "vwap_window": vwap_window,
            "stop_type": stop_type,
            "target_type": target_type,
            "daily_loss_pct": daily_loss_pct
        }
        self.param_cache = ParamCache()

    def get_params(self):
        return self._params

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        self.rsi_period = self._params.get("rsi_period", 14)
        self.vwap_window = self._params.get("vwap_window", None)
        self.stop_type = self._params.get("stop_type", "pivot")
        self.target_type = self._params.get("target_type", "pivot")
        self.daily_loss_pct = self._params.get("daily_loss_pct", 0.01)

    def params_to_str(self, params=None):
        if params is None:
            params = self._params
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def fit(self, train_df, param_grid=None, sampler="bayesian", n_trials=30, n_random_trials=10,
            prev_best_params=None, print_callback=None):
        """
        Param optimization using grid, random, or bayesian search. Caches results.
        """
        # Hash training data to cache params
        h = hashlib.sha256(pd.util.hash_pandas_object(train_df, index=True).values.tobytes()).hexdigest()
        cached = self.param_cache.get(h)
        if cached:
            if print_callback:
                print_callback(0, cached["params"], cached["score"])
            return cached["params"], cached["stats"]

        # Define parameter search space
        import optuna

        def objective(trial):
            rsi_period = trial.suggest_int("rsi_period", 10, 20)
            # vwap_window is not used here (VWAP always cumulative intraday)
            stop_type = "pivot"
            target_type = "pivot"
            daily_loss_pct = trial.suggest_float("daily_loss_pct", 0.005, 0.02)
            self.set_params(
                rsi_period=rsi_period,
                stop_type=stop_type,
                target_type=target_type,
                daily_loss_pct=daily_loss_pct
            )
            signals = self.generate_signals(train_df)
            stats = self.backtest_on_signals(train_df, signals)
            score = stats["win"] / max(1, (stats["win"] + stats["loss"]))  # Win rate
            if print_callback:
                print_callback(trial.number, self.get_params(), score)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        best_params["stop_type"] = "pivot"
        best_params["target_type"] = "pivot"
        self.set_params(**best_params)
        signals = self.generate_signals(train_df)
        best_stats = self.backtest_on_signals(train_df, signals)
        self.param_cache.set(h, {"params": best_params, "score": study.best_value, "stats": best_stats})
        return best_params, best_stats

    def generate_signals(self, df):
        df = df.copy()
        df["vwap"] = calc_vwap(df)
        df["rsi"] = calc_rsi(df["close"], period=self.rsi_period)
        pivots = calc_pivots(df)
        signals = []
        prev_above_vwap = False
        prev_below_vwap = False
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            ts = pd.to_datetime(row["timestamp"], unit="s")
            date = ts.date()
            # Defensive: get only one row from pivots (unique index enforced above)
            pvt_row = pivots.loc[date] if date in pivots.index else None
            # If still a DataFrame or Series with >1 dimension, take first row
            if pvt_row is not None and (isinstance(pvt_row, pd.DataFrame) or (isinstance(pvt_row, pd.Series) and getattr(pvt_row, "ndim", 1) > 1)):
                pvt_row = pvt_row.iloc[0]
            price = float(row["close"])
            vwap = float(row["vwap"])
            rsi = float(row["rsi"])

            # Detect VWAP cross
            crossed_above = float(prev_row["close"]) < float(prev_row["vwap"]) and price > vwap
            crossed_below = float(prev_row["close"]) > float(prev_row["vwap"]) and price < vwap

            if crossed_above and rsi > 50:
                # Buy signal
                stop_loss = np.nan
                target1 = np.nan
                if pvt_row is not None and not pd.isnull(pvt_row["s1"]) and not pd.isnull(pvt_row["r1"]):
                    stop_loss = float(pvt_row["s1"])
                    target1 = float(pvt_row["r1"])
                signals.append({
                    "signal": "BUY",
                    "entry_price": price,
                    "stop_loss": stop_loss,
                    "target1": target1,
                    "timestamp": row["timestamp"]
                })
            elif crossed_below and rsi < 50:
                # Sell signal
                stop_loss = np.nan
                target1 = np.nan
                if pvt_row is not None and not pd.isnull(pvt_row["r1"]) and not pd.isnull(pvt_row["s1"]):
                    stop_loss = float(pvt_row["r1"])
                    target1 = float(pvt_row["s1"])
                signals.append({
                    "signal": "SELL",
                    "entry_price": price,
                    "stop_loss": stop_loss,
                    "target1": target1,
                    "timestamp": row["timestamp"]
                })
            else:
                signals.append({
                    "signal": "",
                    "entry_price": price,
                    "stop_loss": np.nan,
                    "target1": np.nan,
                    "timestamp": row["timestamp"]
                })
        # Padding the first row to align signals with df
        if len(df) > 0:
            first_row = {
                "signal": "",
                "entry_price": float(df.iloc[0]["close"]),
                "stop_loss": np.nan,
                "target1": np.nan,
                "timestamp": df.iloc[0]["timestamp"]
            }
            signals.insert(0, first_row)
        return pd.DataFrame(signals).reset_index(drop=True)

    def backtest_on_signals(self, df, signals, disable_trailing=True):
        # One-trade-at-a-time, EOD exit, daily loss cap, trailing (if enabled)
        trades = []
        equity = []
        capital = 10000
        win = 0
        loss = 0
        gross_profit = 0
        gross_loss = 0
        max_equity = capital
        max_drawdown = 0
        in_trade = False
        entry_idx = None
        entry_price = None
        stop_loss = None
        target1 = None
        direction = None
        entry_time = None
        checkpoint_history = []
        step_num = 0
        trade_entry_date = None
        daily_loss = 0
        current_day = None
        # Defensive: loop over the minimum length
        loop_len = min(len(df), len(signals))
        for i in range(loop_len):
            row = df.iloc[i]
            sig = signals.iloc[i]
            ts = pd.to_datetime(row["timestamp"], unit="s")
            date = ts.date()
            if current_day is None or date != current_day:
                daily_loss = 0
                current_day = date
            if not in_trade and sig["signal"] in ("BUY", "SELL"):
                if daily_loss < -self.daily_loss_pct * capital:
                    continue  # daily loss cap reached
                entry_idx = i
                # Ensure all are float scalars
                entry_price = float(sig["entry_price"]) if not pd.isnull(sig["entry_price"]) else np.nan
                stop_loss = float(sig["stop_loss"]) if not pd.isnull(sig["stop_loss"]) else np.nan
                target1 = float(sig["target1"]) if not pd.isnull(sig["target1"]) else np.nan
                direction = sig["signal"]
                in_trade = True
                entry_time = row["timestamp"]
                trade_entry_date = date
                checkpoint_history = [entry_price]
                step_num = 0
                continue
            if in_trade:
                exit_idx = None
                exit_price = None
                exit_reason = None
                # EOD exit if day changes or after 15:15 IST
                if date != trade_entry_date or ts.time() > pd.to_datetime("15:15:00").time():
                    exit_idx = i
                    exit_price = float(row["close"])
                    exit_reason = "EOD"
                elif direction == "BUY":
                    if not pd.isnull(stop_loss) and float(row["low"]) <= float(stop_loss):
                        exit_idx = i
                        exit_price = float(stop_loss)
                        exit_reason = "Stop"
                    elif not pd.isnull(target1) and float(row["high"]) >= float(target1):
                        exit_idx = i
                        exit_price = float(target1)
                        exit_reason = "Target"
                elif direction == "SELL":
                    if not pd.isnull(stop_loss) and float(row["high"]) >= float(stop_loss):
                        exit_idx = i
                        exit_price = float(stop_loss)
                        exit_reason = "Stop"
                    elif not pd.isnull(target1) and float(row["low"]) <= float(target1):
                        exit_idx = i
                        exit_price = float(target1)
                        exit_reason = "Target"
                # Opposite signal triggers close and new trade
                if sig["signal"] in ("BUY", "SELL") and sig["signal"] != direction:
                    exit_idx = i
                    exit_price = float(row["open"])
                    exit_reason = "Opposite Signal"
                if exit_idx is not None:
                    pl = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
                    capital += pl
                    daily_loss += pl if pl < 0 else 0
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": direction,
                        "exit_reason": exit_reason,
                        "pl": pl,
                        "capital": capital,
                        "entry_time": entry_time,
                        "exit_time": row["timestamp"],
                        "checkpoint_history": checkpoint_history
                    })
                    equity.append(capital)
                    if pl > 0:
                        win += 1
                        gross_profit += pl
                    else:
                        loss += 1
                        gross_loss -= pl
                    if capital > max_equity:
                        max_equity = capital
                    dd = (max_equity - capital)
                    if dd > max_drawdown:
                        max_drawdown = dd
                    in_trade = False
        stats = {
            "win": win,
            "loss": loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "max_drawdown": max_drawdown,
            "trades": trades,
            "equity": equity,
            "sharpe": self.compute_sharpe(equity)
        }
        return stats

    @staticmethod
    def compute_sharpe(equity_curve):
        if not equity_curve or len(equity_curve) < 2:
            return np.nan
        rets = np.diff(equity_curve)
        if rets.std() == 0:
            return 0
        return np.mean(rets) / np.std(rets) * np.sqrt(252)
