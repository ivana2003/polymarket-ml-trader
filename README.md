# 🧠 Polymarket ML Trader
### A machine learning system for prediction market trading on Polymarket

> Built for **Orderflow 001 Hackathon** — 48-hour build sprint  
> Track: **Quantitative Trading**

---

## 📊 Backtest Results

| Metric | Value |
|--------|-------|
| Period | Dec 2022 – Nov 2024 |
| Initial Capital | $1,000 USDC |
| Final Capital | **$4,116 USDC** |
| Total Return | **+311.6%** |
| Total Trades | 6,191 |
| Win Rate | 45.74% |
| Annualized Sharpe | **7.89** |
| Max Drawdown | -$664 |
| Calmar Ratio | **4.69** |
| Profit Factor | **1.66** |

> Backtest on **out-of-sample data only** (GroupKFold by market — no leakage). 2% spread applied to every trade.

---

## 🏗️ System Architecture
```
Polymarket Gamma API          Polymarket CLOB API
       │                              │
       ▼                              ▼
 Resolved Markets            Price History (~1.67h candles)
 (103 markets, $1k+ vol)     (144k candles total)
       │                              │
       └──────────────┬───────────────┘
                      ▼
              Feature Engineering
         (20 features per candle)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Near Model   Mid Model   Far Model
     (< 7 days)  (7-60 days) (> 60 days)
     AUC: 0.777  AUC: 0.682  AUC: 0.655
          │           │
          └─────┬─────┘
                ▼
        Signal Generation
     (threshold: prob > 0.75)
                │
        ┌───────┴───────┐
        ▼               ▼
      LONG            SHORT
  (prob > 0.75)   (prob < 0.25)
        │               │
        └───────┬───────┘
                ▼
     Kelly Position Sizing
     (max 5% capital/trade)
```

---

## 🚀 Quickstart
```bash
git clone https://github.com/ivana2003/polymarket-ml-trader
cd polymarket-ml-trader
pip install -r requirements.txt

# Run live dashboard (uses pre-trained models)
python3 app.py
# Open http://127.0.0.1:8080
```

## 🔄 Retrain models from scratch
```bash
# Fetch data (~15 min)
python3 data/fetch_markets.py
python3 data/fetch_price_history.py

# Train models
python3 models/train.py

# Run backtest
python3 models/backtest.py
```

---

## 🧬 Features (20 per candle)

| Feature | Description |
|---------|-------------|
| `price_ma3/7/14` | Rolling means at multiple windows |
| `price_std7/14` | Local volatility |
| `price_diff1/6/12/24` | Momentum at multiple horizons |
| `price_diff*_norm` | Volatility-normalized momentum |
| `price_relative` | Position in historical range [0,1] |
| `drift_from_start` | Total drift since market open |
| `dist_from_half` | Distance from 0.5 |
| `price_vs_ma` | Price / MA7 ratio |
| `price_zscore14` | Z-score vs 14-period history (mid only) |
| `days_to_close` | Time to resolution |
| `price_x_near/far` | Price × time-bucket interaction |

---

## 🏆 Performance by Bucket

| Bucket | Trades | PnL (USDC) | Win Rate | AUC |
|--------|--------|------------|----------|-----|
| Near (< 7d) | 1,150 | +$1,652 | **52.0%** | 0.777 |
| Mid (7-60d) | 5,041 | +$1,464 | 44.3% | 0.682 |
| **Total** | **6,191** | **+$3,116** | **45.7%** | — |

> Far bucket (> 60 days) excluded: no consistent edge without outlier events.

---

## ⚠️ Honest Assessment

- Backtest on 103 markets — limited universe, US political markets dominant
- Far bucket excluded from live strategy
- Drawdown (-$664) significant relative to initial capital ($1,000)
- Dataset may vary if regenerated via API (market availability changes)

---

*Built by Ivana Crescenzi for Orderflow 001 Hackathon · 2026*
