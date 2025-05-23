# cycore.pyx - Cython-accelerated core for fast trade simulation

import numpy as np
cimport numpy as np

def simulate_trades_core(
    np.ndarray[np.float64_t, ndim=1] arr_high,
    np.ndarray[np.float64_t, ndim=1] arr_low,
    np.ndarray[np.float64_t, ndim=1] arr_close,
    np.ndarray[np.int64_t, ndim=1] arr_timestamp,
    np.ndarray arr_signal,  # unicode type, assume "U8"
    np.ndarray[np.float64_t, ndim=1] arr_entry_price,
    np.ndarray[np.float64_t, ndim=1] arr_stop_loss,
    np.ndarray[np.float64_t, ndim=1] arr_target1,
    double capital_start
):
    # === Trailing/Checkpoint Parameters ===
    cdef double CHECKPOINT_STEP_PCT = 0.5 / 100    # 0.5%
    cdef double THRESHOLDS_PCT[5]
    THRESHOLDS_PCT[0] = 0.7 / 100
    THRESHOLDS_PCT[1] = 1.4 / 100
    THRESHOLDS_PCT[2] = 2.1 / 100
    THRESHOLDS_PCT[3] = 2.8 / 100
    THRESHOLDS_PCT[4] = 3.5 / 100
    cdef int N_THRESH = 5

    cdef Py_ssize_t n = arr_high.shape[0]
    cdef Py_ssize_t i, j, entry_idx, exit_idx
    cdef double capital = capital_start
    cdef double max_equity = capital
    cdef double max_drawdown = 0.0
    cdef double gross_profit = 0.0
    cdef double gross_loss = 0.0
    cdef int win = 0
    cdef int loss = 0
    cdef int in_trade = 0
    cdef double entry_price = 0.0
    cdef double stop_loss = 0.0
    cdef double target1 = 0.0
    cdef double exit_price = 0.0
    cdef double pl = 0.0
    cdef object direction = None
    cdef int step_num
    cdef double current_checkpoint
    cdef double next_threshold
    cdef double threshold_price
    cdef object exit_reason
    trades = []
    equity = [capital]
    for i in range(n):
        sig = arr_signal[i]
        if not in_trade and (sig == "BUY" or sig == "SELL"):
            entry_idx = i
            entry_price = arr_entry_price[i]
            stop_loss = arr_stop_loss[i]
            target1 = arr_target1[i]
            direction = sig
            in_trade = 1
            exit_idx = -1
            exit_price = 0.0
            exit_reason = None

            # --- Trailing/Checkpoint state ---
            step_num = 0
            if direction == "BUY":
                next_threshold = entry_price * (1.0 + THRESHOLDS_PCT[step_num])
                current_checkpoint = entry_price
            else:
                next_threshold = entry_price * (1.0 - THRESHOLDS_PCT[step_num])
                current_checkpoint = entry_price

            for j in range(i+1, n):
                high = arr_high[j]
                low = arr_low[j]
                price = arr_close[j]
                # 1. Stop loss
                if direction == "BUY":
                    if low <= stop_loss:
                        exit_idx = j
                        exit_price = stop_loss
                        exit_reason = "Stop"
                        break
                else:
                    if high >= stop_loss:
                        exit_idx = j
                        exit_price = stop_loss
                        exit_reason = "Stop"
                        break
                # 2. Initial target
                if direction == "BUY":
                    if high >= target1:
                        exit_idx = j
                        exit_price = target1
                        exit_reason = "Target"
                        break
                else:
                    if low <= target1:
                        exit_idx = j
                        exit_price = target1
                        exit_reason = "Target"
                        break
                # 3. Trailing threshold/step logic
                while step_num < N_THRESH:
                    if direction == "BUY":
                        threshold_price = entry_price * (1.0 + THRESHOLDS_PCT[step_num])
                        if high >= threshold_price:
                            step_num += 1
                            # Move checkpoint and stop
                            current_checkpoint = entry_price * (1.0 + CHECKPOINT_STEP_PCT * step_num)
                            stop_loss = current_checkpoint
                            if step_num < N_THRESH:
                                next_threshold = entry_price * (1.0 + THRESHOLDS_PCT[step_num])
                        else:
                            break
                    else:
                        threshold_price = entry_price * (1.0 - THRESHOLDS_PCT[step_num])
                        if low <= threshold_price:
                            step_num += 1
                            current_checkpoint = entry_price * (1.0 - CHECKPOINT_STEP_PCT * step_num)
                            stop_loss = current_checkpoint
                            if step_num < N_THRESH:
                                next_threshold = entry_price * (1.0 - THRESHOLDS_PCT[step_num])
                        else:
                            break
                # 4. Pullback to checkpoint
                if step_num > 0:
                    if direction == "BUY" and low <= current_checkpoint:
                        exit_idx = j
                        exit_price = current_checkpoint
                        exit_reason = f"CheckpointLock (step {step_num})"
                        break
                    elif direction == "SELL" and high >= current_checkpoint:
                        exit_idx = j
                        exit_price = current_checkpoint
                        exit_reason = f"CheckpointLock (step {step_num})"
                        break
            if exit_idx == -1:
                exit_idx = n-1
                exit_price = arr_close[exit_idx]
                exit_reason = "EOD"
            size = 1
            if direction == "BUY":
                pl = (exit_price - entry_price) * size
            else:
                pl = (entry_price - exit_price) * size
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
                "entry_time": int(arr_timestamp[entry_idx]),
                "exit_time": int(arr_timestamp[exit_idx]),
                "checkpoint_steps": step_num,
                "final_checkpoint": float(current_checkpoint)
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
            in_trade = 0
    eq = np.array(equity)
    sharpe = 0.0
    if eq.shape[0] >= 2:
        rets = np.diff(eq)
        std = np.std(rets)
        if std == 0:
            sharpe = 0.0
        else:
            sharpe = np.mean(rets) / std * np.sqrt(252)
    stats = {
        "sharpe": sharpe,
        "profit": capital - capital_start,
        "win": win,
        "loss": loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "num_trades": win + loss
    }
    return stats, trades
