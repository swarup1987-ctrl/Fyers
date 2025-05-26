import numpy as np
import pandas as pd
import itertools
import json
import random
import hashlib
import os
import pickle
import logging
from datetime import datetime, time

# Timezone support for IST conversion
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

def is_entry_allowed(ts_utc):
    dt_utc = datetime.utcfromtimestamp(float(ts_utc))
    try:
        dt_ist = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
    except Exception:
        import pytz
        dt_ist = pytz.UTC.localize(dt_utc).astimezone(IST)
    return (dt_ist.hour < 15) or (dt_ist.hour == 15 and dt_ist.minute <= 15)

def ist_date_and_time(ts_utc):
    dt_utc = datetime.utcfromtimestamp(float(ts_utc))
    try:
        dt_ist = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
    except Exception:
        import pytz
        dt_ist = pytz.UTC.localize(dt_utc).astimezone(IST)
    return dt_ist.date(), dt_ist.time()

class ParamCache:
    def __init__(self, cache_path="corb_opt_cache.pkl"):
        self.cache_path = cache_path
        self._cache = None
        self._dirty = False

    def load(self):
        if self._cache is not None:
            return
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

    def save(self):
        if not self._dirty:
            return
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False
        except Exception as e:
            print(f"ParamCache save error: {e}")

    def get(self, key):
        self.load()
        return self._cache.get(key)

    def set(self, key, value):
        self.load()
        self._cache[key] = value
        self._dirty = True

    def __del__(self):
        self.save()

def hash_train_df(train_df):
    if len(train_df) < 2:
        return None
    start = train_df.iloc[0].get("timestamp", None)
    end = train_df.iloc[-1].get("timestamp", None)
    nrows = len(train_df)
    cols = tuple(train_df.columns)
    h = hashlib.md5(str((start, end, nrows, cols)).encode("utf-8")).hexdigest()
    return h

class CORBStrategy:
    """
    Candle Opening Range Breakout strategy:
    - Uses first 15-min candle as reference range
    - Buy on breakout above high, sell on breakdown below low
    - Stop loss at opposite end, target at 0.5% from entry
    - At most one trade per day
    """
    def __init__(
        self,
        stop_type="range",  # "range" or "mid"
        target_pct=0.5,     # in percent, e.g. 0.5 for 0.5%
    ):
        self.stop_type = stop_type
        self.target_pct = target_pct

    def get_params(self):
        return {
            "stop_type": self.stop_type,
            "target_pct": self.target_pct,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def params_to_str(self, params=None):
        if params is None:
            params = self.get_params()
        return json.dumps(params, sort_keys=True)

    def generate_signals(self, df):
        """
        For each day, mark the first 15-min bar's high/low, then generate one BUY/SELL signal if breakout occurs.
        The signal is marked at the bar where breakout occurs.
        """
        df = df.copy().sort_values("timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        signals = []
        entry_price = []
        stop_loss = []
        target = []
        range_highs = []
        range_lows = []
        trade_type = []
        # Group by date
        for day, day_df in df.groupby("date"):
            if len(day_df) < 2:
                signals.extend([""] * len(day_df))
                entry_price.extend([np.nan] * len(day_df))
                stop_loss.extend([np.nan] * len(day_df))
                target.extend([np.nan] * len(day_df))
                range_highs.extend([np.nan] * len(day_df))
                range_lows.extend([np.nan] * len(day_df))
                trade_type.extend([""] * len(day_df))
                continue
            first_bar = day_df.iloc[0]
            ref_high = first_bar["high"]
            ref_low = first_bar["low"]
            ref_mid = (ref_high + ref_low) / 2
            # Now scan after first bar for breakout
            breakout_idx = None
            breakout_type = None
            for i in range(1, len(day_df)):
                bar = day_df.iloc[i]
                if bar["high"] > ref_high:
                    breakout_idx = i
                    breakout_type = "BUY"
                    break
                elif bar["low"] < ref_low:
                    breakout_idx = i
                    breakout_type = "SELL"
                    break
            # Fill signals for the whole day
            for i in range(len(day_df)):
                if breakout_idx is not None and i == breakout_idx:
                    signals.append(breakout_type)
                    # For realistic backtest, entry is at next bar open, or last bar close if no more bars
                    entry_idx = i+1 if i+1 < len(day_df) else i
                    entry_row = day_df.iloc[entry_idx]
                    entry = entry_row["open"]
                    if self.stop_type == "range":
                        stop = ref_low if breakout_type == "BUY" else ref_high
                    elif self.stop_type == "mid":
                        stop = ref_mid
                    else:
                        stop = ref_low if breakout_type == "BUY" else ref_high
                    tgt = entry * (1 + self.target_pct / 100) if breakout_type == "BUY" else entry * (1 - self.target_pct / 100)
                    entry_price.append(entry)
                    stop_loss.append(stop)
                    target.append(tgt)
                    range_highs.append(ref_high)
                    range_lows.append(ref_low)
                    trade_type.append(breakout_type)
                else:
                    signals.append("")
                    entry_price.append(np.nan)
                    stop_loss.append(np.nan)
                    target.append(np.nan)
                    range_highs.append(ref_high)
                    range_lows.append(ref_low)
                    trade_type.append("")
            # Only one trade per day
        return pd.DataFrame({
            "signal": signals,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "range_high": range_highs,
            "range_low": range_lows,
            "trade_type": trade_type,
        }, index=df.index)

    def backtest_on_signals(self, df, signals, disable_trailing=False):
        """
        For each day, take at most one trade: enter at next bar open after breakout, stop loss at opposite end, target at 0.5%.
        """
        df = df.copy()
        signals = signals.copy()
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        trades = []
        capital = 10000
        equity = [capital]
        win = 0
        loss = 0
        gross_profit = 0
        gross_loss = 0
        max_equity = capital
        max_drawdown = 0

        for day, day_df in df.groupby("date"):
            day_signals = signals.loc[day_df.index]
            # Find the first signal of the day
            idxs = [i for i, s in enumerate(day_signals["signal"]) if s in ["BUY", "SELL"]]
            if not idxs:
                continue
            sig_idx = idxs[0]
            signal = day_signals.iloc[sig_idx]
            # The entry is at the next bar open after breakout, or last bar close
            entry_idx = day_df.index[sig_idx+1] if sig_idx+1 < len(day_df) else day_df.index[sig_idx]
            entry_row = df.loc[entry_idx]
            entry_price = entry_row["open"]
            stop_loss = signal["stop_loss"]
            target = signal["target"]
            direction = signal["signal"]
            trade_entry_ts = entry_row["timestamp"]
            # Now simulate the trade till stop, target, or EOD (15:15)
            exit_idx = None
            exit_price = None
            exit_reason = None
            for j in range(sig_idx+1, len(day_df)):
                row = day_df.iloc[j]
                ts = row["timestamp"]
                ist_date, ist_time = ist_date_and_time(ts)
                high = row["high"]
                low = row["low"]
                # Check EOD
                if ist_time > time(15, 15):
                    exit_idx = day_df.index[j-1] if j > sig_idx+1 else entry_idx
                    exit_price = df.loc[exit_idx, "close"]
                    exit_reason = "EOD"
                    break
                # Stop loss
                if direction == "BUY" and low <= stop_loss:
                    exit_idx = day_df.index[j]
                    exit_price = stop_loss
                    exit_reason = "Stop"
                    break
                elif direction == "SELL" and high >= stop_loss:
                    exit_idx = day_df.index[j]
                    exit_price = stop_loss
                    exit_reason = "Stop"
                    break
                # Target
                if direction == "BUY" and high >= target:
                    exit_idx = day_df.index[j]
                    exit_price = target
                    exit_reason = "Target"
                    break
                elif direction == "SELL" and low <= target:
                    exit_idx = day_df.index[j]
                    exit_price = target
                    exit_reason = "Target"
                    break
            # If trade not closed, close at last bar
            if exit_idx is None:
                exit_idx = day_df.index[-1]
                exit_price = df.loc[exit_idx, "close"]
                exit_reason = "EOD"
            size = 1
            pl = (exit_price - entry_price) * size if direction == "BUY" else (entry_price - exit_price) * size
            capital += pl
            trades.append({
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "direction": direction,
                "exit_reason": exit_reason,
                "pl": float(pl),
                "capital": float(capital),
                "entry_time": int(trade_entry_ts),
                "exit_time": int(df.loc[exit_idx, "timestamp"]),
                "range_high": signal["range_high"],
                "range_low": signal["range_low"],
                "stop_type": self.stop_type,
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

        sharpe = self.compute_sharpe(equity)
        return {
            "sharpe": sharpe,
            "profit": capital - 10000,
            "win": win,
            "loss": loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "num_trades": win + loss,
            "trades": trades,
        }

    @staticmethod
    def compute_sharpe(equity_curve):
        if len(equity_curve) < 2:
            return np.nan
        rets = np.diff(equity_curve)
        if rets.std() == 0:
            return 0
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
        return sharpe

    @staticmethod
    def _evaluate_param_combo(params_dict, train_df_dict, metric="sharpe"):
        train_df = pd.DataFrame(train_df_dict)
        strat = CORBStrategy(**params_dict)
        signals = strat.generate_signals(train_df)
        stats = strat.backtest_on_signals(train_df, signals)
        score = stats.get(metric, 0)
        if np.isnan(score):
            score = -np.inf
        return (params_dict, score, stats)

    def fit(
        self,
        train_df,
        param_grid=None,
        metric='sharpe',
        sampler="bayesian",
        n_trials=10,
        n_random_trials=5,
        sharpe_accept=0.0,
        n_trades_accept=1,
        cache=None,
        cache_path="corb_opt_cache.pkl",
        prev_best_params=None,
        print_callback=None,
        log_file="corb_optuna_trials.log"
    ):
        """
        print_callback: function(trial_number, params, score) for incremental trial logging
        log_file: Optional file path to log all trials
        """
        logger = logging.getLogger("CORBOptunaTrialLogger")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        if param_grid is None:
            param_grid = {
                "stop_type": ["range", "mid"],
                "target_pct": [0.4, 0.5, 0.6, 0.7],
            }

        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*[param_grid[k] for k in param_names]))
        param_dicts = [dict(zip(param_names, vals)) for vals in param_combinations]
        train_df_dict = train_df.to_dict(orient='list')

        # Caching
        if cache is None:
            cache = ParamCache(cache_path)
        train_hash = hash_train_df(train_df)
        cached = cache.get(train_hash)
        if cached:
            cached_params, cached_score, cached_stats = cached
            self.set_params(**cached_params)
            signals = self.generate_signals(train_df)
            stats = self.backtest_on_signals(train_df, signals)
            if (
                stats.get("sharpe", 0) >= sharpe_accept
                and stats.get("num_trades", 0) >= n_trades_accept
            ):
                cache.set(train_hash, (cached_params, stats.get("sharpe", 0), stats))
                cache.save()
                return cached_params, stats

        # If previous best params passed (from last window), try them and enqueue
        best_trial_params = None
        if prev_best_params is not None:
            self.set_params(**prev_best_params)
            signals = self.generate_signals(train_df)
            stats = self.backtest_on_signals(train_df, signals)
            if (
                stats.get("sharpe", 0) >= sharpe_accept
                and stats.get("num_trades", 0) >= n_trades_accept
            ):
                cache.set(train_hash, (prev_best_params, stats.get("sharpe", 0), stats))
                cache.save()
                return prev_best_params, stats
            best_trial_params = prev_best_params

        best_score = -np.inf
        best_params = self.get_params()
        best_stats = None

        if sampler == "grid":
            for idx, params in enumerate(param_dicts):
                _, score, stats = CORBStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[CORB Grid Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
                print(msg)
                logger.info(msg)
                if print_callback:
                    print_callback(idx+1, params, score)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_stats = stats.copy() if stats else None

        elif sampler == "random":
            random_param_dicts = random.sample(param_dicts, min(n_trials, len(param_dicts)))
            for idx, params in enumerate(random_param_dicts):
                _, score, stats = CORBStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[CORB Random Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
                print(msg)
                logger.info(msg)
                if print_callback:
                    print_callback(idx+1, params, score)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_stats = stats.copy() if stats else None

        elif sampler == "bayesian":
            import optuna

            def objective(trial):
                params = {
                    "stop_type": trial.suggest_categorical("stop_type", param_grid["stop_type"]),
                    "target_pct": trial.suggest_categorical("target_pct", param_grid["target_pct"]),
                }
                _, score, stats = CORBStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[CORB Optuna Trial {trial.number}] Params={params} | {metric}: {score:.5f}"
                print(msg)
                logger.info(msg)
                if print_callback:
                    print_callback(trial.number, params, score)
                return score

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(n_startup_trials=n_random_trials))

            if cached:
                study.enqueue_trial(cached[0])
            if best_trial_params is not None:
                study.enqueue_trial(best_trial_params)

            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False
            )
            best_params = study.best_params
            for k in param_grid.keys():
                if k not in best_params:
                    best_params[k] = param_grid[k][0]
            _, best_score, best_stats = CORBStrategy._evaluate_param_combo(best_params, train_df_dict, metric)

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        cache.set(train_hash, (best_params, best_score, best_stats))
        cache.save()
        self.set_params(**best_params)
        return best_params, best_stats
