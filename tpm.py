import numpy as np
import pandas as pd
import json
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

    # Optionally, implement fit() for parameter optimization as in other strategies.
