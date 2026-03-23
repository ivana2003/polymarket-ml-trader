import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CAPITAL, SPREAD, MAX_POS_PCT = 1000, 0.02, 0.05
MIN_PROB, STOP_LOSS, MAX_RETURN = 0.75, -0.05, 0.40

def run_backtest(bucket_results, buckets=["near", "mid"]):
    all_trades = []
    for bucket in buckets:
        preds = bucket_results[bucket]["preds"].copy()
        longs  = preds[preds["prob"] > MIN_PROB].copy()
        shorts = preds[preds["prob"] < (1 - MIN_PROB)].copy()
        longs["side"], shorts["side"] = "long", "short"
        longs["price_return_signed"]  = longs["price_return"]
        shorts["price_return_signed"] = -shorts["price_return"]
        combined = pd.concat([longs, shorts])
        combined = combined[combined["price_return_signed"].abs() <= MAX_RETURN].copy()
        combined["price_return_capped"] = combined["price_return_signed"].clip(lower=STOP_LOSS)
        combined["kelly"] = (2 * combined["prob"].clip(1-MIN_PROB, MIN_PROB) - 1).abs().clip(0, MAX_POS_PCT)
        combined["position_usdc"] = combined["kelly"] * CAPITAL
        combined["pnl_usdc"] = combined["position_usdc"] * (combined["price_return_capped"] - SPREAD)
        combined["time_bucket"] = bucket
        all_trades.append(combined)
    trades = pd.concat(all_trades).sort_values("timestamp").reset_index(drop=True)
    trades["cumulative_pnl"] = trades["pnl_usdc"].cumsum()
    trades["capital"]        = CAPITAL + trades["cumulative_pnl"]
    trades["drawdown"]       = trades["capital"] - trades["capital"].cummax()
    return trades

def print_report(trades):
    gross_profit = trades[trades["pnl_usdc"] > 0]["pnl_usdc"].sum()
    gross_loss   = trades[trades["pnl_usdc"] < 0]["pnl_usdc"].abs().sum()
    sharpe_ann   = (trades["pnl_usdc"].mean() / trades["pnl_usdc"].std()) * np.sqrt(len(trades) / 2)
    print("=" * 52)
    print("   POLYMARKET ML TRADER — BACKTEST REPORT")
    print("=" * 52)
    print(f"Initial Capital:   ${CAPITAL:,.0f} USDC")
    print(f"Final Capital:     ${trades['capital'].iloc[-1]:,.2f} USDC")
    print(f"Total Return:      {(trades['capital'].iloc[-1]/CAPITAL - 1)*100:.1f}%")
    print(f"Total Trades:      {len(trades):,}")
    print(f"Win Rate:          {(trades['pnl_usdc'] > 0).mean():.2%}")
    print(f"Sharpe (ann):      {sharpe_ann:.4f}")
    print(f"Max Drawdown:      ${trades['drawdown'].min():,.2f}")
    print(f"Calmar Ratio:      {trades['pnl_usdc'].sum()/abs(trades['drawdown'].min()):.4f}")
    print(f"Profit Factor:     {gross_profit/gross_loss:.4f}")
    print("=" * 52)
