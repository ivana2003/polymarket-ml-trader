import requests, json, pandas as pd
from tqdm import tqdm

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

def fetch_resolved_markets(max_offset=5000, limit=100):
    all_resolved = []
    offset = 0
    while offset < max_offset:
        params = {"closed": "true", "limit": limit, "offset": offset}
        resp = requests.get(GAMMA_URL, params=params)
        batch = resp.json()
        if not batch:
            break
        for m in batch:
            try:
                prices = [float(p) for p in json.loads(m["outcomePrices"])]
                if len(prices) == 2 and max(prices) > 0.99 and min(prices) < 0.01:
                    all_resolved.append(m)
            except:
                continue
        offset += limit
        if offset % 500 == 0:
            print(f"Offset {offset}: {len(all_resolved)} resolved so far")
    return all_resolved

def save_markets(markets, output_path="data/resolved_markets.csv"):
    campi = ["id","question","category","closedTime","volume","clobTokenIds"]
    df = pd.DataFrame(markets)[campi].copy()
    df["volume"] = df["volume"].astype(float)
    df["closedTime"] = pd.to_datetime(df["closedTime"], format="mixed", utc=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} markets to {output_path}")
    return df

if __name__ == "__main__":
    markets = fetch_resolved_markets()
    save_markets(markets)
