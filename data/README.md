# Data

## Files (not tracked by git)
- `resolved_markets.csv` — 4,751 resolved markets from Gamma API
- `price_history_resolved.csv` — 144k price candles from CLOB API

## Regenerate
```bash
python fetch_markets.py
python fetch_price_history.py
```
