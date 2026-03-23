import requests, json, pandas as pd, numpy as np
from tqdm import tqdm

CLOB_URL = "https://clob.polymarket.com/prices-history"

def get_price_history(token_id, end_time, days_back=365, window_days=7):
    all_history = []
    end_ts = int(end_time.timestamp())
    window_sec = window_days * 24 * 3600
    current_end = end_ts
    current_start = current_end - window_sec
    cutoff = end_ts - (days_back * 24 * 3600)
    while current_end > cutoff:
        resp = requests.get(CLOB_URL, params={
            "market": token_id, "startTs": current_start,
            "endTs": current_end, "fidelity": 100
        })
        if resp.status_code == 200:
            all_history = resp.json().get("history", []) + all_history
        current_end = current_start
        current_start = current_end - window_sec
    return all_history

def fetch_all_histories(markets_csv="data/resolved_markets.csv",
                        min_volume=1000, min_date="2023-01-01",
                        output_path="data/price_history_resolved.csv"):
    df_markets = pd.read_csv(markets_csv)
    df_markets["closedTime"] = pd.to_datetime(df_markets["closedTime"], format="mixed", utc=True)
    df_markets["volume"] = df_markets["volume"].astype(float)
    df_filtered = df_markets[
        (df_markets["closedTime"] >= min_date) &
        (df_markets["volume"] > min_volume)
    ]
    print(f"Markets to download: {len(df_filtered)}")
    results = []
    for m in tqdm(df_filtered.itertuples(), total=len(df_filtered)):
        try:
            token_yes = json.loads(m.clobTokenIds)[0]
            history = get_price_history(token_yes, m.closedTime)
            for row in history:
                results.append({
                    "market_id": m.id, "question": m.question,
                    "category": m.category, "token_id": token_yes,
                    "closedTime": m.closedTime, "t": row["t"], "price": row["p"]
                })
        except Exception as e:
            print(f"Skip {m.id}: {e}")
    df_out = pd.DataFrame(results)
    df_out["timestamp"] = pd.to_datetime(df_out["t"], unit="s", utc=True)
    df_out["days_to_close"] = (
        pd.to_datetime(df_out["closedTime"], utc=True) - df_out["timestamp"]
    ).dt.total_seconds() / 86400
    final_prices = df_out.groupby("market_id")["price"].last().rename("final_price")
    df_out = df_out.merge(final_prices, on="market_id")
    df_out["outcome"] = (df_out["final_price"] > 0.5).astype(int)
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} rows to {output_path}")
    return df_out

if __name__ == "__main__":
    fetch_all_histories()
