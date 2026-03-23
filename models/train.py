import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import pickle, os, sys
sys.path.append(os.path.dirname(__file__))
from features import build_features, build_target, FEATURE_SETS

def train_bucket_models(data_path="data/price_history_resolved.csv",
                        model_dir="models/saved", n_splits=5):
    print("Loading data...")
    df = pd.read_csv(data_path)
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  format="mixed", utc=True)
    df["closedTime"] = pd.to_datetime(df["closedTime"], format="mixed", utc=True)
    df = build_features(df)
    df = build_target(df)
    df_model = df.dropna(subset=["future_price", "target"]).copy()
    df_model["target"] = df_model["target"].astype(int)
    print(f"Dataset: {df_model.shape} — {df_model['market_id'].nunique()} markets")
    os.makedirs(model_dir, exist_ok=True)
    results = {}
    for bucket in ["near", "mid", "far"]:
        df_b = df_model[df_model["time_bucket"] == bucket].copy()
        feats = FEATURE_SETS[bucket]
        X, y, groups = df_b[feats], df_b["target"], df_b["market_id"]
        k = min(n_splits, groups.nunique())
        gkf = GroupKFold(n_splits=k)
        auc_scores, all_preds = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            auc_scores.append(roc_auc_score(y.iloc[test_idx], y_prob))
            test_df = df_b.iloc[test_idx].copy()
            test_df["prob"] = y_prob
            all_preds.append(test_df)
        final_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        final_model.fit(X, y)
        with open(f"{model_dir}/model_{bucket}.pkl", "wb") as f:
            pickle.dump(final_model, f)
        results[bucket] = {
            "auc_mean": np.mean(auc_scores), "auc_std": np.std(auc_scores),
            "preds": pd.concat(all_preds), "model": final_model
        }
        print(f"[{bucket:4s}] AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f} | n={len(df_b)}")
    return results

if __name__ == "__main__":
    train_bucket_models()
