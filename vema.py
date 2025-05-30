import numpy as np
import pandas as pd
import hashlib
import pickle
import os

class ParamCache:
    """Caches best parameters for a given hash of training data."""
    def __init__(self, fname="vema_param_cache.pkl"):
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

def hash_train_df(train_df):
    if len(train_df) < 2:
        return None
    return hashlib.sha256(pd.util.hash_pandas_object(train_df, index=True).values.tobytes()).hexdigest()

class VEMAStrategy:
    """
    EMA9/EMA21 cross strategy with volume filter for 15-min charts.
    - Buy: EMA9 crosses above EMA21, and volume > moving average volume.
    - Sell: EMA9 crosses below EMA21, and volume > moving average volume.
    - Stop loss and target optimized.
    """

    def __init__(self,
                 stop_loss_pct=0.5,   # Percentage (e.g., 0.5 for 0.5%)
                 target_pct=0.8,      # Percentage (e.g., 0.8 for 0.8%)
                 vol_window=20,       # Volume moving average window
                 daily_loss_cap=0.01  # 1% daily loss cap (optional)
                 ):
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.vol_window = vol_window
        self.daily_loss_cap = daily_loss_cap
        self.param_cache = ParamCache()
        self._params = {
            "stop_loss_pct": stop_loss_pct,
            "target_pct": target_pct,
            "vol_window": vol_window,
            "daily_loss_cap": daily_loss_cap
        }

    def get_params(self):
        return self._params.copy()

    def set_params(self, **kwargs):
        self._params.update(kwargs)
        self.stop_loss_pct = self._params.get("stop_loss_pct", 0.5)
        self.target_pct = self._params.get("target_pct", 0.8)
        self.vol_window = self._params.get("vol_window", 20)
        self.daily_loss_cap = self._params.get("daily_loss_cap", 0.01)

    def params_to_str(self, params=None):
        if params is None:
            params = self._params
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def fit(self, train_df, param_grid=None, sampler="bayesian", n_trials=8, n_random_trials=3,
            prev_best_params=None, print_callback=None):
        """
        Param optimization for stop_loss_pct and target_pct.
        Keeps search space small for efficiency.
        """
        h = hash_train_df(train_df)
        cached = self.param_cache.get(h)
        if cached:
            if print_callback:
                print_callback(0, cached["params"], cached["score"])
            return cached["params"], cached["stats"]

        import optuna

        # Small, reasonable grid to keep Optuna light
        stop_loss_space = [0.4, 0.5, 0.6]   # %
        target_space = [0.7, 0.8, 0.9]      # %
        vol_window_space = [15, 20, 25]     # optional

        def objective(trial):
            stop_loss_pct = trial.suggest_categorical("stop_loss_pct", stop_loss_space)
            target_pct = trial.suggest_categorical("target_pct", target_space)
            vol_window = trial.suggest_categorical("vol_window", vol_window_space)
            self.set_params(stop_loss_pct=stop_loss_pct, target_pct=target_pct, vol_window=vol_window)
            signals = self.generate_signals(train_df)
            stats = self.backtest_on_signals(train_df, signals)
            # Use win rate for score, but you can use Sharpe etc. if preferred
            score = stats.get("win", 0) / max(1, stats.get("win", 0) + stats.get("loss", 0))
            if print_callback:
                print_callback(trial.number, self.get_params(), score)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        self.set_params(**best_params)
        signals = self.generate_signals(train_df)
        best_stats = self.backtest_on_signals(train_df, signals)
        self.param_cache.set(h, {"params": best_params, "score": study.best_value, "stats": best_stats})
        return best_params, best_stats

    def generate_signals(self, df):
        df = df.copy()
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["vol_ma"] = df["volume"].rolling(self.vol_window, min_periods=1).mean()
        # Cross above: prev ema9 <= prev ema21 and now ema9 > ema21
        cross_above = (df["ema9"].shift(1) <= df["ema21"].shift(1)) & (df["ema9"] > df["ema21"])
        cross_below = (df["ema9"].shift(1) >= df["ema21"].shift(1)) & (df["ema9"] < df["ema21"])
        above_avg_vol = df["volume"] > df["vol_ma"]

        signals = []
        for i in range(len(df)):
            sig = ""
            entry_price = np.nan
            stop_loss = np.nan
            target1 = np.nan

            if cross_above.iloc[i] and above_avg_vol.iloc[i]:
                sig = "BUY"
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                target1 = entry_price * (1 + self.target_pct / 100)
            elif cross_below.iloc[i] and above_avg_vol.iloc[i]:
                sig = "SELL"
                entry_price = df["close"].iloc[i]
                stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
                target1 = entry_price * (1 - self.target_pct / 100)

            signals.append({
                "signal": sig,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target1": target1,
                "timestamp": df["timestamp"].iloc[i]
            })
        return pd.DataFrame(signals)

    def backtest_on_signals(self, df, signals, disable_trailing=True):
        # One-trade-at-a-time, EOD exit, daily loss cap, no trailing logic (for simplicity)
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
                if daily_loss < -self.daily_loss_cap * capital:
                    continue  # daily loss cap reached
                entry_idx = i
                entry_price = float(sig["entry_price"]) if not pd.isnull(sig["entry_price"]) else np.nan
                stop_loss = float(sig["stop_loss"]) if not pd.isnull(sig["stop_loss"]) else np.nan
                target1 = float(sig["target1"]) if not pd.isnull(sig["target1"]) else np.nan
                direction = sig["signal"]
                in_trade = True
                entry_time = row["timestamp"]
                trade_entry_date = date
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
                        "checkpoint_history": [],  # no trailing here
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
