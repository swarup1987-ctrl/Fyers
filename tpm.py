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
    def __init__(self, cache_path="tpm_opt_cache.pkl"):
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

class TPMStrategy:
    def __init__(
        self,
        ema_period=20,
        rsi_period=9,
        vma_period=50,
        profit_target_pct=0.9,      # default 0.9%, user can override
        stop_loss_pct=0.45,         # default 0.45%, user can override
        use_trailing=True,
        trailing_target_pct=None,   # trailing logic, optional (if None, disables)
        trailing_stop_pct=None,
    ):
        # Core params
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.vma_period = vma_period
        self.profit_target_pct = profit_target_pct / 100 if profit_target_pct else 0.009
        self.stop_loss_pct = stop_loss_pct / 100 if stop_loss_pct else 0.0045
        # Trailing: if both are set, use trailing; else, just fixed stops
        self.use_trailing = use_trailing and trailing_target_pct is not None and trailing_stop_pct is not None
        self.trailing_target_pct = trailing_target_pct / 100 if trailing_target_pct else None
        self.trailing_stop_pct = trailing_stop_pct / 100 if trailing_stop_pct else None

    def get_params(self):
        return {
            "ema_period": self.ema_period,
            "rsi_period": self.rsi_period,
            "vma_period": self.vma_period,
            "profit_target_pct": self.profit_target_pct * 100,
            "stop_loss_pct": self.stop_loss_pct * 100,
            "use_trailing": self.use_trailing,
            "trailing_target_pct": self.trailing_target_pct * 100 if self.trailing_target_pct is not None else None,
            "trailing_stop_pct": self.trailing_stop_pct * 100 if self.trailing_stop_pct is not None else None,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k.endswith('pct') and v is not None:
                    setattr(self, k, v / 100 if v > 1 else v)
                else:
                    setattr(self, k, v)
        # Recompute use_trailing
        self.use_trailing = self.trailing_target_pct is not None and self.trailing_stop_pct is not None

    def params_to_str(self, params=None):
        if params is None:
            params = self.get_params()
        return json.dumps(params, sort_keys=True)

    def calculate_indicators(self, df):
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        df['ema20_slope'] = df['ema20'].diff()
        df['rsi9'] = self._rsi(df['close'], self.rsi_period)
        df['vma50'] = df['volume'].rolling(self.vma_period, min_periods=1).mean()
        # Swing high/low: local maxima/minima for support/resistance (look-back 5)
        df['swing_high'] = df['high'].rolling(5, center=True, min_periods=1).max()
        df['swing_low'] = df['low'].rolling(5, center=True, min_periods=1).min()
        return df

    def _rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _is_ema_trending(self, ema_slope, threshold=1e-3):
        # "Trending" if absolute slope > threshold (not flat)
        return abs(ema_slope) > threshold

    def _bullish_candle(self, row):
        return row['close'] > row['open'] and row['low'] == min(row['open'], row['close'], row['low'])

    def _bearish_candle(self, row):
        return row['close'] < row['open'] and row['high'] == max(row['open'], row['close'], row['high'])

    def generate_signals(self, df):
        df = self.calculate_indicators(df)
        signals = []
        entry_price = []
        stop_loss = []
        target = []
        swing_lv = []
        for i, row in df.iterrows():
            # Trend filter
            trending = self._is_ema_trending(row['ema20_slope'])
            bullish_bias = row['close'] > row['ema20'] and trending
            bearish_bias = row['close'] < row['ema20'] and trending

            # Momentum triggers
            rsi_cross_up = (
                (row['rsi9'] > 30)
                and (df['rsi9'].iloc[i-1] if i > 0 else 100) <= 30
            )
            rsi_cross_dn = (
                (row['rsi9'] < 70)
                and (df['rsi9'].iloc[i-1] if i > 0 else 0) >= 70
            )

            # Volume confirmation
            vol_surge = row['volume'] > row['vma50']

            # Support/resistance
            near_support = abs(row['low'] - row['swing_low']) < 1e-6
            near_resist = abs(row['high'] - row['swing_high']) < 1e-6

            # Candle structure
            is_bull_candle = self._bullish_candle(row)
            is_bear_candle = self._bearish_candle(row)

            # Entry signal
            sig = ""
            if bullish_bias and rsi_cross_up and vol_surge and is_bull_candle and near_support:
                sig = "BUY"
            elif bearish_bias and rsi_cross_dn and vol_surge and is_bear_candle and near_resist:
                sig = "SELL"
            signals.append(sig)

            # Entry logic: next bar open, but for signal table, set at this bar (entry is handled in backtest)
            entry_price.append(row['open'] if sig else np.nan)

            # Stop loss and target
            if sig == "BUY":
                stp = row['open'] * (1 - self.stop_loss_pct)
                tgt = row['open'] * (1 + self.profit_target_pct)
            elif sig == "SELL":
                stp = row['open'] * (1 + self.stop_loss_pct)
                tgt = row['open'] * (1 - self.profit_target_pct)
            else:
                stp = tgt = np.nan
            stop_loss.append(stp)
            target.append(tgt)
            swing_lv.append(row['swing_low'] if sig == "BUY" else row['swing_high'] if sig == "SELL" else np.nan)
        return pd.DataFrame({
            'signal': signals,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'swing': swing_lv
        }, index=df.index)

    def backtest_on_signals(self, df, signals, disable_trailing=False):
        """
        Backtest logic with support for trailing and risk management.
        """
        capital = 10000
        equity = [capital]
        trades = []
        win, loss = 0, 0
        gross_profit, gross_loss = 0, 0
        max_equity = capital
        max_drawdown = 0
        daily_loss = 0
        trading_date = None
        daily_loss_cap_pct = 0.01  # 1%

        in_trade = False
        entry_idx = None
        entry_price = None
        stop_loss = None
        target = None
        direction = None
        trade_date = None
        step_num = 0
        trailing_active = self.use_trailing and not disable_trailing

        i = 0
        while i < len(df):
            row = df.iloc[i]
            sig = signals.iloc[i]
            bar_date, bar_time = ist_date_and_time(row["timestamp"])
            # Reset daily loss on new day
            if trading_date is not None and bar_date != trading_date:
                trading_date = bar_date
                daily_loss = 0
            if not in_trade:
                # Check daily loss cap
                if daily_loss <= -daily_loss_cap_pct * capital:
                    i += 1
                    continue
                # Entry only at signal, and must be allowed by time
                if sig['signal'] in ("BUY", "SELL") and is_entry_allowed(row["timestamp"]):
                    entry_idx = i + 1 if i + 1 < len(df) else i  # enter next bar open
                    if entry_idx >= len(df):
                        break
                    entry_row = df.iloc[entry_idx]
                    entry_price = entry_row['open']
                    stop_loss = (
                        entry_price * (1 - self.stop_loss_pct) if sig['signal'] == "BUY"
                        else entry_price * (1 + self.stop_loss_pct)
                    )
                    target = (
                        entry_price * (1 + self.profit_target_pct) if sig['signal'] == "BUY"
                        else entry_price * (1 - self.profit_target_pct)
                    )
                    direction = sig['signal']
                    trade_date = bar_date
                    in_trade = True
                    step_num = 0
                    trail_price = entry_price
                    i = entry_idx + 1
                    continue
            else:
                exit_idx = None
                exit_price = None
                exit_reason = None
                # Trade management loop
                for j in range(i, len(df)):
                    rowj = df.iloc[j]
                    ts = rowj["timestamp"]
                    barj_date, barj_time = ist_date_and_time(ts)
                    # EOD/time exit
                    if barj_date != trade_date or barj_time > time(15, 15):
                        exit_idx = j - 1 if j > i else i
                        exit_price = df.loc[exit_idx, "close"]
                        exit_reason = "EOD"
                        i = exit_idx + 1
                        break
                    high = rowj["high"]
                    low = rowj["low"]
                    # Stop loss
                    if direction == "BUY" and low <= stop_loss:
                        exit_idx = j
                        exit_price = stop_loss
                        exit_reason = "Stop"
                        i = j + 1
                        break
                    elif direction == "SELL" and high >= stop_loss:
                        exit_idx = j
                        exit_price = stop_loss
                        exit_reason = "Stop"
                        i = j + 1
                        break
                    # Target
                    if direction == "BUY" and high >= target:
                        exit_idx = j
                        exit_price = target
                        exit_reason = "Target"
                        i = j + 1
                        break
                    elif direction == "SELL" and low <= target:
                        exit_idx = j
                        exit_price = target
                        exit_reason = "Target"
                        i = j + 1
                        break
                    # Trailing (if enabled)
                    if trailing_active:
                        # Move stop up if price moves by trailing_target_pct from entry
                        if direction == "BUY":
                            move_up = entry_price * (1 + self.trailing_target_pct) if self.trailing_target_pct else None
                            if self.trailing_target_pct and high >= move_up:
                                step_num += 1
                                stop_loss = max(stop_loss, entry_price * (1 + self.trailing_stop_pct * step_num))
                        elif direction == "SELL":
                            move_dn = entry_price * (1 - self.trailing_target_pct) if self.trailing_target_pct else None
                            if self.trailing_target_pct and low <= move_dn:
                                step_num += 1
                                stop_loss = min(stop_loss, entry_price * (1 - self.trailing_stop_pct * step_num))
                        # If price falls back to moved stop
                        if direction == "BUY" and low <= stop_loss:
                            exit_idx = j
                            exit_price = stop_loss
                            exit_reason = f"TrailStop"
                            i = j + 1
                            break
                        elif direction == "SELL" and high >= stop_loss:
                            exit_idx = j
                            exit_price = stop_loss
                            exit_reason = f"TrailStop"
                            i = j + 1
                            break
                # If trade is still open at end, close at last bar
                if exit_idx is None and in_trade:
                    exit_idx = len(df) - 1
                    exit_price = df.loc[exit_idx, "close"]
                    exit_reason = "EOD"
                    i = exit_idx + 1
                # P&L calc
                size = 1
                pl = (exit_price - entry_price) * size if direction == "BUY" else (entry_price - exit_price) * size
                capital += pl
                daily_loss += pl
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
        strat = TPMStrategy(**params_dict)
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
        cache_path="tpm_opt_cache.pkl",
        prev_best_params=None,
        print_callback=None,
        log_file="tpm_optuna_trials.log"
    ):
        """
        print_callback: function(trial_number, params, score) for incremental trial logging
        log_file: Optional file path to log all trials
        """
        import concurrent.futures

        logger = logging.getLogger("TPMOptunaTrialLogger")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        if param_grid is None:
            param_grid = {
                "ema_period": [10, 15, 20, 25],
                "rsi_period": [7, 9, 12],
                "vma_period": [30, 40, 50, 60],
                "profit_target_pct": [0.7, 0.8, 0.9, 1.0, 1.1],  # Percent
                "stop_loss_pct": [0.35, 0.4, 0.45, 0.5, 0.55],    # Percent
                "use_trailing": [False, True],
                "trailing_target_pct": [None, 0.5, 0.7, 0.9],     # Percent
                "trailing_stop_pct": [None, 0.25, 0.35, 0.45],    # Percent
            }

        # Remove trailing params if use_trailing is False for a combo
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*[param_grid[k] for k in param_names]))
        param_dicts = []
        for vals in param_combinations:
            params = dict(zip(param_names, vals))
            if not params["use_trailing"]:
                params["trailing_target_pct"] = None
                params["trailing_stop_pct"] = None
            param_dicts.append(params)
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
                _, score, stats = TPMStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[TPM Grid Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
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
                _, score, stats = TPMStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[TPM Random Trial {idx+1}] Params={params} | {metric}: {score:.5f}"
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
                    "ema_period": trial.suggest_categorical("ema_period", param_grid["ema_period"]),
                    "rsi_period": trial.suggest_categorical("rsi_period", param_grid["rsi_period"]),
                    "vma_period": trial.suggest_categorical("vma_period", param_grid["vma_period"]),
                    "profit_target_pct": trial.suggest_categorical("profit_target_pct", param_grid["profit_target_pct"]),
                    "stop_loss_pct": trial.suggest_categorical("stop_loss_pct", param_grid["stop_loss_pct"]),
                    "use_trailing": trial.suggest_categorical("use_trailing", param_grid["use_trailing"]),
                    "trailing_target_pct": trial.suggest_categorical("trailing_target_pct", param_grid["trailing_target_pct"]),
                    "trailing_stop_pct": trial.suggest_categorical("trailing_stop_pct", param_grid["trailing_stop_pct"]),
                }
                if not params["use_trailing"]:
                    params["trailing_target_pct"] = None
                    params["trailing_stop_pct"] = None
                _, score, stats = TPMStrategy._evaluate_param_combo(params, train_df_dict, metric)
                msg = f"[TPM Optuna Trial {trial.number}] Params={params} | {metric}: {score:.5f}"
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
            if not best_params["use_trailing"]:
                best_params["trailing_target_pct"] = None
                best_params["trailing_stop_pct"] = None
            _, best_score, best_stats = TPMStrategy._evaluate_param_combo(best_params, train_df_dict, metric)

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        cache.set(train_hash, (best_params, best_score, best_stats))
        cache.save()
        self.set_params(**best_params)
        return best_params, best_stats
