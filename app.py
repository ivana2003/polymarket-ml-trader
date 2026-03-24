
import requests, json, numpy as np, pandas as pd
import os
import pickle
from models.features import build_features
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR = "models/saved"

def load_models():
    models = {}
    for bucket in ["near", "mid", "far"]:
        path = os.path.join(MODEL_DIR, f"model_{bucket}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[bucket] = pickle.load(f)
    return models

MODELS = load_models()
print("Loaded models:", MODELS.keys())

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com/markets"

def get_open_markets(limit=20):
    """Fetch active open markets with high volume."""
    params = {"active": "true", "closed": "false", "limit": limit, 
              "order": "volume24hr", "ascending": "false"}
    resp = requests.get(GAMMA_URL, params=params, timeout=10)
    markets = resp.json()
    result = []
    for m in markets:
        try:
            prices = [float(p) for p in json.loads(m["outcomePrices"])]
            if len(prices) == 2 and m.get("clobTokenIds"):
                token_ids = json.loads(m["clobTokenIds"])
                result.append({
                    "market_id": m["id"],
                    "question": m["question"],
                    "category": m.get("category", "N/A"),
                    "volume24hr": float(m.get("volume24hr", 0)),
                    "price_yes": prices[0],
                    "price_no": prices[1],
                    "token_yes": token_ids[0],
                    "spread": float(m.get("spread", 0.02)),
                    "end_date": m.get("endDate"),
                })
        except:
            continue
    return result

def get_recent_prices(token_id, hours=72):
    """Get recent price history for a token."""
    import time
    end_ts = int(time.time())
    start_ts = end_ts - hours * 3600
    resp = requests.get(f"{CLOB_URL}/prices-history", params={
        "market": token_id, "startTs": start_ts, 
        "endTs": end_ts, "fidelity": 100
    }, timeout=10)
    if resp.status_code == 200:
        return resp.json().get("history", [])
    return []

def build_live_features(market_id, end_date_str, prices_raw):
    if len(prices_raw) < 30:
        return None

    end_dt = pd.to_datetime(end_date_str, utc=True, errors="coerce")
    if pd.isna(end_dt):
        return None

    rows = []
    for p in prices_raw:
        ts = pd.to_datetime(p["t"], unit="s", utc=True, errors="coerce")
        price = float(p["p"])
        days_to_close = (end_dt - ts).total_seconds() / (24 * 3600)

        rows.append({
            "market_id": market_id,
            "timestamp": ts,
            "price": price,
            "days_to_close": days_to_close
        })

    df_live = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    df_live = build_features(df_live)

    if df_live.empty:
        return None

    return df_live.iloc[-1].copy()

def ml_signal_from_history(market):
    print("ML SIGNAL FUNCTION HIT")
    prices_raw = get_recent_prices(market["token_yes"], hours=72)
    if len(prices_raw) < 30:
        return None

    last_row = build_live_features(
        market_id=market["market_id"],
        end_date_str=market["end_date"],
        prices_raw=prices_raw
    )
    if last_row is None:
        return None

    bucket = str(last_row["time_bucket"])
    if bucket not in MODELS:
        return None

    model_bundle = MODELS[bucket]
    model = model_bundle["model"]
    feats = model_bundle["features"]

    # costruisco l'input live con solo le feature richieste
    X_live = pd.DataFrame([last_row[feats]])

    prob_up = model.predict_proba(X_live)[0, 1]

    if prob_up > 0.75:
        signal = "BUY"
        confidence = prob_up
    elif prob_up < 0.25:
        signal = "SELL"
        confidence = 1 - prob_up
    else:
        signal = "HOLD"
        confidence = max(prob_up, 1 - prob_up)

    return {
        "signal": signal,
        "confidence": round(float(confidence), 3),
        "prob_up": round(float(prob_up), 3),
        "bucket": bucket,
        "current_price": round(float(last_row["price"]), 4),
        "days_to_close": round(float(last_row["days_to_close"]), 2),
        "diff24": round(float(last_row.get("price_diff24", 0)), 4),
        "zscore": round(float(last_row.get("price_zscore14", 0)), 3),
        "n_candles": len(prices_raw),
    }




@app.route("/api/signals")
def get_signals():
    """Main endpoint: returns live signals for top markets."""
    try:
        markets = get_open_markets(limit=15)
        signals = []
        for m in markets[:10]:
            sig = ml_signal_from_history(m)
            if sig:
                signals.append({**m, **sig})
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return jsonify({"status": "ok", "signals": signals, "count": len(signals)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/market/<token_id>")
def get_market_detail(token_id):
    """Price history for a specific token."""
    prices = get_recent_prices(token_id, hours=168)
    return jsonify({"history": prices})

@app.route("/")
def index():
    return open("dashboard/live.html", encoding="utf-8").read()

if __name__ == "__main__":
    print("Starting Polymarket ML Trader server...")
    print("Dashboard: http://localhost:5000")
    app.run(debug=False, port=8080)
