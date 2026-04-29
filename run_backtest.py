import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle, os, sys, argparse
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from features import build_features, build_target, FEATURE_SETS

CAPITAL      = 1000
SPREAD       = 0.02
MAX_POS_PCT  = 0.05
MIN_PROB     = 0.75
STOP_LOSS    = -0.05
MAX_RETURN   = 0.40
TRAIN_MONTHS = 3

def load_and_prepare(data_path):
    print(f"[1/5] Caricamento dati da: {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  format="mixed", utc=True)
    df["closedTime"] = pd.to_datetime(df["closedTime"], format="mixed", utc=True)
    df = build_features(df)
    df = build_target(df)
    df = df.dropna(subset=["future_price", "target"]).copy()
    df["target"] = df["target"].astype(int)
    df["month"]  = df["timestamp"].dt.to_period("M")
    print(f"    {len(df):,} righe — {df['market_id'].nunique()} mercati — {df['month'].nunique()} mesi")
    return df

def walk_forward_backtest(df):
    months = sorted(df["month"].unique())
    if len(months) <= TRAIN_MONTHS:
        raise ValueError(f"Servono almeno {TRAIN_MONTHS + 1} mesi di dati (trovati: {len(months)})")
    print(f"[2/5] Walk-forward su {len(months)} mesi (training: {TRAIN_MONTHS}, test: {len(months)-TRAIN_MONTHS})")
    all_preds, auc_log = [], []
    for i, test_month in enumerate(months[TRAIN_MONTHS:], start=TRAIN_MONTHS):
        train_months = months[:i]
        train_df = df[df["month"].isin(train_months)]
        test_df  = df[df["month"] == test_month].copy()
        month_preds = []
        for bucket in ["near", "mid"]:
            feats = FEATURE_SETS[bucket]
            tr = train_df[train_df["time_bucket"] == bucket]
            te = test_df[test_df["time_bucket"] == bucket].copy()
            if len(tr) < 50 or len(te) == 0:
                continue
            model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            model.fit(tr[feats], tr["target"])
            te["prob"] = model.predict_proba(te[feats])[:, 1]
            if len(te["target"].unique()) > 1:
                auc_log.append({"month": str(test_month), "bucket": bucket,
                                "auc": roc_auc_score(te["target"], te["prob"])})
            month_preds.append(te)
        if month_preds:
            all_preds.append(pd.concat(month_preds))
        print(f"    Mese {i-TRAIN_MONTHS+1}/{len(months)-TRAIN_MONTHS}: {test_month}")
    return pd.concat(all_preds).reset_index(drop=True), pd.DataFrame(auc_log)

def compute_pnl(preds):
    print("[3/5] Calcolo PnL...")
    longs  = preds[preds["prob"] >  MIN_PROB].copy()
    shorts = preds[preds["prob"] < (1 - MIN_PROB)].copy()
    longs["side"], shorts["side"] = "long", "short"
    longs["price_return_signed"]  =  longs["price_return"]
    shorts["price_return_signed"] = -shorts["price_return"]
    trades = pd.concat([longs, shorts])
    trades = trades[trades["price_return_signed"].abs() <= MAX_RETURN].copy()
    trades["price_return_capped"] = trades["price_return_signed"].clip(lower=STOP_LOSS)
    trades["kelly"]               = (2 * trades["prob"].clip(1-MIN_PROB, MIN_PROB) - 1).abs().clip(0, MAX_POS_PCT)
    trades["position_usdc"]       = trades["kelly"] * CAPITAL
    trades["pnl_usdc"]            = trades["position_usdc"] * (trades["price_return_capped"] - SPREAD)
    trades = trades.sort_values("timestamp").reset_index(drop=True)
    trades["cumulative_pnl_usdc"] = trades["pnl_usdc"].cumsum()
    trades["capital"]             = CAPITAL + trades["cumulative_pnl_usdc"]
    trades["drawdown_usdc"]       = trades["capital"] - trades["capital"].cummax()
    trades["month"]               = trades["timestamp"].dt.to_period("M").astype(str)
    return trades

def print_report(trades, auc_df):
    gross_profit = trades[trades["pnl_usdc"] > 0]["pnl_usdc"].sum()
    gross_loss   = trades[trades["pnl_usdc"] < 0]["pnl_usdc"].abs().sum()
    sharpe_ann   = (trades["pnl_usdc"].mean() / trades["pnl_usdc"].std()) * np.sqrt(len(trades) / 2)
    max_dd       = trades["drawdown_usdc"].min()
    print("\n" + "=" * 54)
    print("   POLYMARKET ML TRADER — BACKTEST OOS REPORT")
    print("=" * 54)
    print(f"Capital iniziale:  ${CAPITAL:,.0f} USDC")
    print(f"Capital finale:    ${trades['capital'].iloc[-1]:,.2f} USDC")
    print(f"Return totale:     {(trades['capital'].iloc[-1]/CAPITAL - 1)*100:.1f}%")
    print(f"N. trade:          {len(trades):,}")
    print(f"Win rate:          {(trades['pnl_usdc'] > 0).mean():.2%}")
    print(f"Sharpe (ann):      {sharpe_ann:.4f}")
    print(f"Max Drawdown:      ${max_dd:,.2f}")
    calmar = trades["pnl_usdc"].sum() / abs(max_dd) if max_dd != 0 else float("nan")
    pf     = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"Calmar Ratio:      {calmar:.4f}")
    print(f"Profit Factor:     {pf:.4f}")
    if not auc_df.empty:
        print(f"AUC medio OOS:     {auc_df['auc'].mean():.4f} ± {auc_df['auc'].std():.4f}")
    print("\nPnL mensile:")
    for m, pnl in trades.groupby("month")["pnl_usdc"].sum().items():
        bar  = "▓" * int(abs(pnl) / 20)
        sign = "+" if pnl >= 0 else "-"
        icon = "✅" if pnl >= 0 else "❌"
        print(f"  {m}:  {sign}${abs(pnl):6.1f} {bar} {icon}")
    print("=" * 54)

def plot_equity(trades, out_path):
    print(f"[4/5] Salvataggio grafico: {out_path}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Polymarket ML Trader — Equity Curve (OOS Walk-Forward)", fontsize=13)
    ax1.plot(trades.index, trades["capital"], color="#2196F3", linewidth=1.5)
    ax1.axhline(CAPITAL, color="gray", linestyle="--", linewidth=0.8)
    ax1.fill_between(trades.index, trades["capital"], CAPITAL,
                     where=trades["capital"] >= CAPITAL, alpha=0.1, color="green")
    ax1.fill_between(trades.index, trades["capital"], CAPITAL,
                     where=trades["capital"] <  CAPITAL, alpha=0.1, color="red")
    ax1.set_ylabel("USDC")
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(trades.index, trades["drawdown_usdc"], 0, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown USDC")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_results(trades, out_path):
    print(f"[5/5] Salvataggio CSV: {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"    Salvato: {len(trades):,} trade")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="data/price_history_resolved.csv")
    parser.add_argument("--out-csv",  default="results/backtest_oos.csv")
    parser.add_argument("--out-plot", default="results/equity_curve_oos.png")
    args = parser.parse_args()
    df            = load_and_prepare(args.data)
    preds, auc_df = walk_forward_backtest(df)
    trades        = compute_pnl(preds)
    print_report(trades, auc_df)
    plot_equity(trades, args.out_plot)
    save_results(trades, args.out_csv)
    print("\nFatto! Trovi i risultati in results/backtest_oos.csv e results/equity_curve_oos.png")
