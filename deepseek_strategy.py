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

# --- Try to import Cython core, fallback if not available ---
try:
    import cycore
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# Timezone support for IST conversion
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

def is_entry_allowed(ts_utc):
    """
    Restrict trade entries after 15:15 IST.
    ts_utc: timestamp (seconds since epoch, UTC)
    Returns True only if bar's IST time is <= 15:15
    """
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
    def __init__(self, cache_path="opt_cache.pkl"):
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

class Intraday15MinStrategy:
    def __init__(
        self,
        bb_period=20,
        bb_std_dev=2,
        rsi_period=8,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_period=14,
        use_cycore=True,
        initial_threshold_pct=0.7,   # now user-configurable, in percent (can be None)
        initial_checkpoint_pct=0.5   # now user-configurable, in percent (can be None)
    ):
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.use_cycore = use_cycore and CYTHON_AVAILABLE
        # Accept None for trailing params
        self.initial_threshold_pct = (initial_threshold_pct / 100) if initial_threshold_pct is not None else None
        self.initial_checkpoint_pct = (initial_checkpoint_pct / 100) if initial_checkpoint_pct is not None else None

    def get_params(self):
        return {
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'atr_period': self.atr_period,
            'initial_threshold_pct': (self.initial_threshold_pct * 100) if self.initial_threshold_pct is not None else None,
            'initial_checkpoint_pct': (self.initial_checkpoint_pct * 100) if self.initial_checkpoint_pct is not None else None,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k in ["initial_threshold_pct", "initial_checkpoint_pct"]:
                    setattr(self, k, (v / 100) if (v is not None and v > 1) else (v if v is not None else None))
                else:
                    setattr(self, k, v)

    def params_to_str(self, params=None):
        if params is None:
            params = self.get_params()
        return json.dumps(params, sort_keys=True)

    def save_params(self, filepath, params=None):
        if params is None:
            params = self.get_params()
        with open(filepath, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)

    def load_params(self, filepath):
        with open(filepath, "r") as f:
            params = json.load(f)
        self.set_params(**params)

    def calculate_indicators(self, df):
        df = df.copy()
        df['middle_band'] = df['close'].rolling(self.bb_period, min_periods=1).mean()
        rolling_std = df['close'].rolling(self.bb_period, min_periods=1).std()
        df['upper_band'] = df['middle_band'] + rolling_std * self.bb_std_dev
        df['lower_band'] = df['middle_band'] - rolling_std * self.bb_std_dev
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(self.rsi_period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['ema_fast'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd_line'].ewm(span=self.macd_signal, adjust=False).mean()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = df['tr'].rolling(self.atr_period, min_periods=1).mean()
        df.drop(['cum_tp_vol', 'cum_vol'], axis=1, inplace=True)
        return df

    def generate_signals(self, df):
        df = self.calculate_indicators(df.copy())
        macd_cross_up = ((df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) & (df['macd_line'] > df['macd_signal'])) | \
                        ((df['macd_line'].shift(2) <= df['macd_signal'].shift(2)) & (df['macd_line'].shift(1) > df['macd_signal'].shift(1)))
        macd_cross_down = ((df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) & (df['macd_line'] < df['macd_signal'])) | \
                          ((df['macd_line'].shift(2) >= df['macd_signal'].shift(2)) & (df['macd_line'].shift(1) < df['macd_signal'].shift(1)))
        buy_checks = [
            df['close'] > df['vwap'],
            macd_cross_up,
            df['rsi'] < 80,
            df['close'] <= df['lower_band'] + 0.25 * df['atr']
        ]
        sell_checks = [
            df['close'] < df['vwap'],
            macd_cross_down,
            df['rsi'] > 22,
            df['close'] >= df['upper_band'] - 0.25 * df['atr']
        ]
        buy_condition = sum(buy_checks) >= 3
        sell_condition = sum(sell_checks) >= 3

        signals = np.where(buy_condition, 'BUY', np.where(sell_condition, 'SELL', ''))
        entry_price = np.where(signals != '', df['close'], np.nan)
        last3_low = df['low'].rolling(3, min_periods=1).min()
        last3_high = df['high'].rolling(3, min_periods=1).max()
        buy_stop = np.minimum(df['close'] - 1.5 * df['atr'], last3_low - 0.02 * df['close'])
        sell_stop = np.maximum(df['close'] + 1.5 * df['atr'], last3_high + 0.02 * df['close'])
        stop_loss = np.where(
            signals == 'BUY', buy_stop,
            np.where(signals == 'SELL', sell_stop, np.nan)
        )
        buy_target = np.maximum(
            df['close'] + 1.5 * (df['close'] - buy_stop),
            np.minimum(df['upper_band'], df['vwap'] + df['atr'])
        )
        sell_target = np.minimum(
            df['close'] - 1.5 * (sell_stop - df['close']),
            np.maximum(df['lower_band'], df['vwap'] - df['atr'])
        )
        target1 = np.where(
            signals == 'BUY', buy_target,
            np.where(signals == 'SELL', sell_target, np.nan)
        )
        target2 = np.where(
            signals == 'BUY', df['upper_band'],
            np.where(signals == 'SELL', df['lower_band'], np.nan)
        )
        signals_df = pd.DataFrame({
            'signal': signals,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2
        }, index=df.index)
        return signals_df

    def backtest_on_signals(self, df, signals, disable_trailing=False):
        """
        Restrict trade entries after 15:15 IST.
        Restrict trade exits at or before 15:15 IST (force EOD exit at 15:15 IST).
        Only one trade at a time. If a new, opposing signal is generated, exit current trade and enter new one.
        """
        n_steps = 5
        if disable_trailing or self.initial_threshold_pct is None or self.initial_checkpoint_pct is None:
            use_trailing = False
            CHECKPOINT_STEP_PCT = 0.0
            THRESHOLDS_PCT = [0.0] * n_steps
        else:
            use_trailing = True
            CHECKPOINT_STEP_PCT = self.initial_checkpoint_pct
            THRESHOLDS_PCT = [self.initial_threshold_pct * (i+1) for i in range(n_steps)]

        trades = []
        capital = 10000
        equity = [capital]
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
        checkpoint_history = []
        step_num = 0
        trade_entry_ist_date = None

        i = 0
        while i < len(df):
            sig = signals.iloc[i]
            row = df.iloc[i]
            bar_ist_date, bar_ist_time = ist_date_and_time(row["timestamp"])

            # --- If NOT in a trade, check for new signal ---
            if not in_trade and sig["signal"] in ("BUY", "SELL") and is_entry_allowed(row["timestamp"]):
                # Open new trade
                entry_idx = i
                entry_price = sig["entry_price"]
                stop_loss = sig["stop_loss"]
                target1 = sig["target1"]
                direction = sig["signal"]
                in_trade = True
                trade_entry_ist_date = bar_ist_date
                checkpoint_history = [entry_price]
                step_num = 0
                next_threshold = entry_price * (1 + THRESHOLDS_PCT[step_num]) if direction == "BUY" else entry_price * (1 - THRESHOLDS_PCT[step_num])
                current_checkpoint = entry_price
                i += 1
                continue

            # --- If in a trade, simulate trade bar by bar ---
            if in_trade:
                exit_idx = None
                exit_price = None
                exit_reason = None

                # Check for stop/target/threshold/15:15/EOD/opposite signal
                for j in range(i, len(df)):
                    rowj = df.iloc[j]
                    high = rowj["high"]
                    low = rowj["low"]
                    price = rowj["close"]
                    ts = rowj["timestamp"]
                    ist_date, ist_time = ist_date_and_time(ts)
                    sigj = signals.iloc[j]

                    # 1. EOD/15:15 IST restriction: force exit at 15:15 IST or if day changes
                    if ist_date != trade_entry_ist_date or ist_time > time(15, 15):
                        # Exit at previous bar (last bar <= 15:15 IST)
                        exit_idx = j - 1 if j > i else i
                        exit_price = df.loc[exit_idx, "close"]
                        exit_reason = "EOD"
                        i = exit_idx + 1
                        break

                    # 2. Stop loss
                    if direction == "BUY":
                        if low <= stop_loss:
                            exit_idx = j
                            exit_price = stop_loss
                            exit_reason = "Stop"
                            i = j + 1
                            break
                    else:
                        if high >= stop_loss:
                            exit_idx = j
                            exit_price = stop_loss
                            exit_reason = "Stop"
                            i = j + 1
                            break

                    # 3. Initial target
                    if direction == "BUY":
                        if high >= target1:
                            exit_idx = j
                            exit_price = target1
                            exit_reason = "Target"
                            i = j + 1
                            break
                    else:
                        if low <= target1:
                            exit_idx = j
                            exit_price = target1
                            exit_reason = "Target"
                            i = j + 1
                            break

                    # 4. Trailing logic
                    if use_trailing:
                        while step_num < len(THRESHOLDS_PCT):
                            threshold_price = entry_price * (1 + THRESHOLDS_PCT[step_num]) if direction == "BUY" else entry_price * (1 - THRESHOLDS_PCT[step_num])
                            if (direction == "BUY" and high >= threshold_price) or (direction == "SELL" and low <= threshold_price):
                                step_num += 1
                                new_checkpoint = entry_price * (1 + CHECKPOINT_STEP_PCT * step_num) if direction == "BUY" else entry_price * (1 - CHECKPOINT_STEP_PCT * step_num)
                                current_checkpoint = new_checkpoint
                                checkpoint_history.append(new_checkpoint)
                                stop_loss = new_checkpoint
                                if step_num < len(THRESHOLDS_PCT):
                                    next_threshold = entry_price * (1 + THRESHOLDS_PCT[step_num]) if direction == "BUY" else entry_price * (1 - THRESHOLDS_PCT[step_num])
                            else:
                                break
                        if step_num > 0:
                            if direction == "BUY" and low <= current_checkpoint:
                                exit_idx = j
                                exit_price = current_checkpoint
                                exit_reason = f"CheckpointLock (step {step_num})"
                                i = j + 1
                                break
                            elif direction == "SELL" and high >= current_checkpoint:
                                exit_idx = j
                                exit_price = current_checkpoint
                                exit_reason = f"CheckpointLock (step {step_num})"
                                i = j + 1
                                break

                    # 5. Opposite signal: exit immediately and enter new trade
                    if sigj["signal"] in ("BUY", "SELL") and sigj["signal"] != direction and is_entry_allowed(ts):
                        exit_idx = j
                        exit_price = rowj["open"]  # Use open price of this bar for slippage realism
                        exit_reason = "Opposite Signal"
                        # Will enter new trade in main loop at i = j
                        i = j
                        break

                    # else continue loop

                # If we exited by break, record the trade
                if exit_idx is not None:
                    size = 1
                    pl = (exit_price - entry_price) * size if direction == "BUY" else (entry_price - exit_price) * size
                    capital += pl
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": direction,
                        "exit_reason": exit_reason,
                        "pl": pl,
                        "capital": capital,
                        "entry_time": df.loc[entry_idx, "timestamp"],
                        "exit_time": df.loc[exit_idx, "timestamp"],
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
                    # If exited by opposite signal, the main loop will check and start new trade here
                    continue

                # If we reached the end of the data, close trade at last bar
                if in_trade:
                    exit_idx = len(df) - 1
                    exit_price = df.loc[exit_idx, "close"]
                    exit_reason = "EOD"
                    size = 1
                    pl = (exit_price - entry_price) * size if direction == "BUY" else (entry_price - exit_price) * size
                    capital += pl
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": direction,
                        "exit_reason": exit_reason,
                        "pl": pl,
                        "capital": capital,
                        "entry_time": df.loc[entry_idx, "timestamp"],
                        "exit_time": df.loc[exit_idx, "timestamp"],
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
                    break
            i += 1

        sharpe = self.compute_sharpe(equity)
        return {
            "sharpe": sharpe,
            "profit": capital - 10000,
            "win": win,
            "loss": loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "num_trades": win + loss,
            "trades": trades
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
        strat = Intraday15MinStrategy(**params_dict)
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
        n_trials=50,
        n_random_trials=20,
        sharpe_accept=0.8,
        n_trades_accept=5,
        cache=None,
        cache_path="opt_cache.pkl",
        prev_best_params=None,
        print_callback=None,
        log_file="optuna_trials.log"
    ):
        """
        print_callback: function(trial_number, params, score) for incremental trial logging
        log_file: Optional file path to log all trials
        """
        import concurrent.futures

        logger = logging.getLogger("OptunaTrialLogger")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        if param_grid is None:
            param_grid = {
                "bb_period": [12, 16, 20, 24],
                "bb_std_dev": [1.5, 2.0, 2.5],
                "rsi_period": [6, 8, 12],
                "macd_fast": [8, 10, 12],
                "macd_slow": [18, 22, 26, 30],
                "macd_signal": [7, 9, 11],
                "atr_period": [10, 14, 18],
                # Optionally, could grid search threshold/checkpoint too
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
            # Try previous best params on current train set
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
            # Else, will optimize and update cache

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
                _, score, stats = Intraday15MinStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[Grid Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
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
                _, score, stats = Intraday15MinStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[Random Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
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
                    "bb_period": trial.suggest_categorical("bb_period", param_grid["bb_period"]),
                    "bb_std_dev": trial.suggest_categorical("bb_std_dev", param_grid["bb_std_dev"]),
                    "rsi_period": trial.suggest_categorical("rsi_period", param_grid["rsi_period"]),
                    "macd_fast": trial.suggest_categorical("macd_fast", param_grid["macd_fast"]),
                    "macd_slow": trial.suggest_categorical("macd_slow", param_grid["macd_slow"]),
                    "macd_signal": trial.suggest_categorical("macd_signal", param_grid["macd_signal"]),
                    "atr_period": trial.suggest_categorical("atr_period", param_grid["atr_period"]),
                    # You could add threshold/checkpoint to Optuna search if desired
                }
                _, score, stats = Intraday15MinStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[Optuna Trial {trial.number}] Params={params} | {metric}: {score:.5f}"
                print(msg)
                logger.info(msg)
                if print_callback:
                    print_callback(trial.number, params, score)
                return score

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(n_startup_trials=n_random_trials))

            # Enqueue cache and prev_best
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
            _, best_score, best_stats = Intraday15MinStrategy._evaluate_param_combo(best_params, train_df_dict, metric)

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        cache.set(train_hash, (best_params, best_score, best_stats))
        cache.save()
        self.set_params(**best_params)
        return best_params, best_stats
