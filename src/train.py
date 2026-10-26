"""Train baselines and AI models, evaluate them on the unseen test period, save models and predictions.

Models
- Baseline: building weekly profile (average of the same hour-of-week in training)
- Baseline: same hour last week (day-ahead)
- Ridge regression (weather model)
- Random Forest (weather model)
- LightGBM weather model: weather + calendar + building only  -> used for anomalies and M&V
- LightGBM day-ahead model: + consumption up to the previous day -> used for operational forecasts
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, MODELS, REPORTS, SEED, TRAIN_END
from src.features import (CALENDAR, CATEGORICAL, DAY_AHEAD_FEATURES, PROFILE, WEATHER,
                          WEATHER_MODEL_FEATURES, build_features)
from src.metrics import per_building, summary

VALID_START = "2017-05-01"  # last two months of training, used only to choose the number of trees

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,  # smaller, more regular trees won on the validation months (reports/tuning_validation.csv)
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "seed": SEED,
    "verbose": -1,
    "num_threads": 0,
}


def best_rounds(valid_df: pd.DataFrame, features: list[str], name: str) -> int:
    """Number of trees, found by early stopping on May-Jun 2017.

    valid_df is built with building profiles learned only from data before May 2017,
    so the validation months are truly unseen (no target leakage).
    """
    fit = valid_df[valid_df["timestamp"] < VALID_START]
    val = valid_df[(valid_df["timestamp"] >= VALID_START) & (valid_df["timestamp"] < TRAIN_END)]
    dfit = lgb.Dataset(fit[features], np.log1p(fit["kwh"]), categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(val[features], np.log1p(val["kwh"]), reference=dfit)
    probe = lgb.train(LGB_PARAMS, dfit, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = int(probe.best_iteration * 1.1)  # a little more data in the final fit
    print(f"  {name}: best iteration {probe.best_iteration} on validation, final model uses {rounds} trees")
    return rounds


def train_lgbm(train: pd.DataFrame, features: list[str], name: str, rounds: int) -> lgb.Booster:
    t = time.time()
    full = lgb.Dataset(train[features], np.log1p(train["kwh"]), categorical_feature=CATEGORICAL)
    model = lgb.train(LGB_PARAMS, full, num_boost_round=rounds)
    model.save_model(str(MODELS / f"{name}.txt"))
    print(f"  {name}: trained on the full training period ({time.time() - t:.0f}s)")
    return model


def numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric view for scikit-learn models: categories become integer codes / one-hot."""
    x = df[CALENDAR + WEATHER + PROFILE].copy()
    x = pd.concat([x, pd.get_dummies(df["primary_use"], prefix="use", dtype=np.float32)], axis=1)
    return x


def main() -> None:
    t0 = time.time()
    raw = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    print("Choosing the number of trees on leak-free validation months ...")
    valid_df, _ = build_features(raw, train_end=VALID_START)
    rounds = {"lgbm_weather": best_rounds(valid_df, WEATHER_MODEL_FEATURES, "lgbm_weather"),
              "lgbm_day_ahead": best_rounds(valid_df, DAY_AHEAD_FEATURES, "lgbm_day_ahead")}
    del valid_df
    print("Building features ...")
    df, profiles = build_features(raw)
    joblib.dump(profiles, MODELS / "building_profiles.joblib")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    print(f"  train {len(train):,} rows | test {len(test):,} rows | {df['building_id'].nunique()} buildings")

    preds = pd.DataFrame(index=df.index)

    print("Baselines ...")
    preds["baseline_profile"] = np.expm1(df["bld_how_mean_log"])
    preds["baseline_last_week"] = np.expm1(df["lag_168h"]).fillna(preds["baseline_profile"])

    print("Ridge regression ...")
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(numeric_matrix(train), np.log1p(train["kwh"]))
    preds["ridge"] = np.expm1(ridge.predict(numeric_matrix(df)))

    print("Random Forest (600k-row sample) ...")
    sample = train.sample(600_000, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=60, max_depth=24, min_samples_leaf=5, max_features=0.6,
                               n_jobs=-1, random_state=SEED)
    rf.fit(numeric_matrix(sample), np.log1p(sample["kwh"]))
    preds["random_forest"] = np.nan
    preds.loc[test.index, "random_forest"] = np.expm1(rf.predict(numeric_matrix(test)))
    del rf

    print("LightGBM weather model ...")
    weather_model = train_lgbm(train, WEATHER_MODEL_FEATURES, "lgbm_weather", rounds["lgbm_weather"])
    preds["lgbm_weather"] = np.expm1(weather_model.predict(df[WEATHER_MODEL_FEATURES]))

    print("LightGBM day-ahead model ...")
    day_ahead_model = train_lgbm(train, DAY_AHEAD_FEATURES, "lgbm_day_ahead", rounds["lgbm_day_ahead"])
    preds["lgbm_day_ahead"] = np.expm1(day_ahead_model.predict(df[DAY_AHEAD_FEATURES]))

    # Ensemble of the two strongest weather-only models
    preds["ensemble_weather"] = np.nan
    preds.loc[test.index, "ensemble_weather"] = (0.7 * preds.loc[test.index, "lgbm_weather"]
                                                 + 0.3 * preds.loc[test.index, "random_forest"])

    out = pd.concat([df[["timestamp", "building_id", "site", "primary_use", "kwh", "temperature", "humidity", "split"]],
                     preds.astype(np.float32)], axis=1)
    out.to_parquet(DATA_PROCESSED / "predictions.parquet", index=False)

    print("Evaluating on the test period (2017-07-01 .. 2017-12-31) ...")
    test_out = out[out["split"] == "test"]
    model_info = {
        "baseline_profile": ("Baseline: weekly profile", "weather"),
        "ridge": ("Ridge regression", "weather"),
        "random_forest": ("Random Forest", "weather"),
        "lgbm_weather": ("LightGBM (weather model)", "weather"),
        "ensemble_weather": ("Ensemble LightGBM + RF", "weather"),
        "baseline_last_week": ("Baseline: same hour last week", "day-ahead"),
        "lgbm_day_ahead": ("LightGBM (day-ahead model)", "day-ahead"),
    }
    results = []
    for col, (label, task) in model_info.items():
        results.append({"model": label, "column": col, "task": task, **summary(test_out, col)})
    results = pd.DataFrame(results)
    results.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(results.drop(columns="column").to_string(index=False))

    for col in ["lgbm_weather", "lgbm_day_ahead"]:
        pb = per_building(test_out, col)
        pb.to_csv(REPORTS / f"per_building_{col}.csv", index=False)
        by_use = pb.groupby("primary_use", observed=True).agg(
            buildings=("building_id", "size"), median_cv_rmse=("cv_rmse", "median"),
            median_abs_nmbe=("nmbe", lambda s: s.abs().median())).round(2).sort_values("buildings", ascending=False)
        by_use.to_csv(REPORTS / f"by_primary_use_{col}.csv")

    info = {"features_weather_model": WEATHER_MODEL_FEATURES, "features_day_ahead": DAY_AHEAD_FEATURES,
            "train_rows": len(train), "test_rows": len(test), "buildings": int(df["building_id"].nunique()),
            "runtime_minutes": round((time.time() - t0) / 60, 1)}
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Done in {info['runtime_minutes']} min")


if __name__ == "__main__":
    main()
