
import requests, json, numpy as np, pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
                    "spread": float(m.get("spread", 0.02))
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

def compute_signal(prices_raw):
    """Compute trading signal from price history."""
    if len(prices_raw) < 10:
        return None
    prices = [p["p"] for p in prices_raw]
    p = np.array(prices)
    current = p[-1]
    ma7  = p[-min(7,  len(p)):].mean()
    ma14 = p[-min(14, len(p)):].mean()
    std7 = p[-min(7,  len(p)):].std() + 1e-9
    diff1  = p[-1] - p[-2]  if len(p) > 1  else 0
    diff24 = p[-1] - p[-25] if len(p) > 24 else 0
    drift  = p[-1] - p[0]
    p_min, p_max = p.min(), p.max()
    price_rel = (current - p_min) / (p_max - p_min + 1e-9)
    zscore = (current - ma14) / std7

    # Rule-based signal (simplified model logic)
    score = 0.5
    # Momentum signals
    if diff24 > 0.02: score += 0.08
    if diff24 < -0.02: score -= 0.08
    if diff1 > 0.01: score += 0.04
    if diff1 < -0.01: score -= 0.04
    # Mean reversion signals
    if zscore > 2.0: score -= 0.10  # overbought -> short
    if zscore < -2.0: score += 0.10  # oversold -> long
    # Trend signals
    if current > ma7 > ma14: score += 0.06
    if current < ma7 < ma14: score -= 0.06
    # Position signals
    if price_rel < 0.15: score += 0.05  # near historical low
    if price_rel > 0.85: score -= 0.05  # near historical high

    score = max(0.05, min(0.95, score))

    if score > 0.65:
        signal = "BUY"
        confidence = score
    elif score < 0.35:
        signal = "SELL"
        confidence = 1 - score
    else:
        signal = "HOLD"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "score": round(score, 3),
        "current_price": round(current, 4),
        "ma7": round(ma7, 4),
        "zscore": round(zscore, 3),
        "diff24": round(diff24, 4),
        "drift": round(drift, 4),
        "n_candles": len(prices)
    }

@app.route("/api/signals")
def get_signals():
    """Main endpoint: returns live signals for top markets."""
    try:
        markets = get_open_markets(limit=15)
        signals = []
        for m in markets[:10]:
            prices = get_recent_prices(m["token_yes"], hours=72)
            sig = compute_signal(prices)
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
    return open("dashboard/live.html").read()

if __name__ == "__main__":
    print("Starting Polymarket ML Trader server...")
    print("Dashboard: http://localhost:5000")
    app.run(debug=False, port=8080)
