import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    grp = df.groupby("market_id")["price"]
    df["price_ma3"]   = grp.transform(lambda x: x.rolling(3,  min_periods=1).mean())
    df["price_ma7"]   = grp.transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["price_ma14"]  = grp.transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["price_std7"]  = grp.transform(lambda x: x.rolling(7,  min_periods=1).std().fillna(0))
    df["price_std14"] = grp.transform(lambda x: x.rolling(14, min_periods=1).std().fillna(0))
    df["price_diff1"]  = grp.transform(lambda x: x.diff(1).fillna(0))
    df["price_diff6"]  = grp.transform(lambda x: x.diff(6).fillna(0))
    df["price_diff12"] = grp.transform(lambda x: x.diff(12).fillna(0))
    df["price_diff24"] = grp.transform(lambda x: x.diff(24).fillna(0))
    for lag in [1, 6, 24]:
        df[f"price_diff{lag}_norm"] = (df[f"price_diff{lag}"] / (df["price_std7"] + 1e-9)).clip(-5, 5)
    df["dist_from_half"]   = (df["price"] - 0.5).abs()
    df["drift_from_start"] = grp.transform(lambda x: x - x.iloc[0])
    df["price_min"]        = grp.transform("min")
    df["price_max"]        = grp.transform("max")
    df["price_relative"]   = (df["price"] - df["price_min"]) / (df["price_max"] - df["price_min"] + 1e-9)
    df["price_vs_ma"]      = df["price"] / (df["price_ma7"] + 1e-9)
    roll_mean = grp.transform(lambda x: x.rolling(14, min_periods=3).mean())
    roll_std  = grp.transform(lambda x: x.rolling(14, min_periods=3).std())
    df["price_zscore14"] = ((df["price"] - roll_mean) / (roll_std + 1e-9)).clip(-4, 4).fillna(0)
    df["time_bucket"] = pd.cut(df["days_to_close"], bins=[0, 7, 60, 9999], labels=["near", "mid", "far"])
    df["price_x_near"] = df["price"] * (df["time_bucket"] == "near").astype(int)
    df["price_x_far"]  = df["price"] * (df["time_bucket"] == "far").astype(int)
    return df

def build_target(df, n=24, threshold=0.01):
    grp = df.groupby("market_id")["price"]
    df["future_price"] = grp.transform(lambda x: x.shift(-n))
    df["price_return"] = df["future_price"] - df["price"]
    df["target"] = np.where(df["price_return"] > threshold, 1,
                   np.where(df["price_return"] < -threshold, 0, np.nan))
    return df

FEATURE_SETS = {
    "near": ["price","price_ma3","price_ma7","price_ma14","price_std7","price_std14",
             "price_diff1","price_diff6","price_diff12","price_diff24",
             "price_diff1_norm","price_diff6_norm","price_diff24_norm",
             "dist_from_half","drift_from_start","price_relative",
             "price_vs_ma","days_to_close","price_x_near","price_x_far"],
    "mid":  ["price","price_ma3","price_ma7","price_ma14","price_std7","price_std14",
             "price_diff1","price_diff6","price_diff12","price_diff24",
             "price_diff1_norm","price_diff6_norm","price_diff24_norm",
             "dist_from_half","drift_from_start","price_relative",
             "price_vs_ma","days_to_close","price_x_near","price_x_far","price_zscore14"],
    "far":  ["price","price_ma3","price_ma7","price_ma14","price_std7","price_std14",
             "price_diff1","price_diff6","price_diff12","price_diff24",
             "price_diff1_norm","price_diff6_norm","price_diff24_norm",
             "dist_from_half","drift_from_start","price_relative",
             "price_vs_ma","days_to_close","price_x_near","price_x_far"]
}
