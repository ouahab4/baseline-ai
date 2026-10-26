"""Compare feature sets and LightGBM settings on a validation period (May-Jun 2017).

The test period (Jul-Dec 2017) is not used here, so the final test score stays honest.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, REPORTS, TRAIN_END
from src.features import CATEGORICAL, DAY_AHEAD_FEATURES, LAGS, WEATHER_MODEL_FEATURES, build_features
from src.metrics import summary
from src.train import LGB_PARAMS, VALID_START

OLD_PROFILE = ["bld_mean_log", "bld_how_mean_log", "bld_month_mean_log"]
RECENT = ["bld_recent_mean_log", "bld_recent_how_log"]
# Experiments as run before the final settings were chosen (results in reports/tuning_validation.csv)
WITH_DOY = WEATHER_MODEL_FEATURES[:6] + ["dayofyear"] + WEATHER_MODEL_FEATURES[6:]
NO_RECENT = [f for f in WITH_DOY if f not in RECENT]
NO_DOY = WEATHER_MODEL_FEATURES
ORIGINAL = {"num_leaves": 255, "min_data_in_leaf": 100, "learning_rate": 0.08}
SMALL = {"num_leaves": 63, "min_data_in_leaf": 500, "learning_rate": 0.05}

VARIANTS = {
    "weather: original features": (NO_RECENT, ORIGINAL),
    "weather: + recent level": (WITH_DOY, ORIGINAL),
    "weather: + recent level - dayofyear": (NO_DOY, ORIGINAL),
    "weather: + recent - dayofyear, smaller trees": (NO_DOY, SMALL),
    "day-ahead: original features": (NO_RECENT + LAGS, ORIGINAL),
    "day-ahead: + recent - dayofyear": (NO_DOY + LAGS, ORIGINAL),
    "day-ahead: + recent - dayofyear, smaller trees": (NO_DOY + LAGS, SMALL),
}


def main() -> None:
    df, _ = build_features(pd.read_parquet(DATA_PROCESSED / "dataset.parquet"), train_end=VALID_START)
    fit = df[df["timestamp"] < VALID_START]
    val = df[(df["timestamp"] >= VALID_START) & (df["timestamp"] < TRAIN_END)].copy()
    y_fit, y_val = np.log1p(fit["kwh"]), np.log1p(val["kwh"])
    rows = []
    for name, (feats, extra) in VARIANTS.items():
        t = time.time()
        params = {**LGB_PARAMS, **extra}
        dfit = lgb.Dataset(fit[feats], y_fit, categorical_feature=CATEGORICAL)
        dval = lgb.Dataset(val[feats], y_val, reference=dfit)
        m = lgb.train(params, dfit, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        val["p"] = np.expm1(m.predict(val[feats], num_iteration=m.best_iteration))
        s = summary(val, "p")
        rows.append({"variant": name, "best_iteration": m.best_iteration, **s})
        print(f"{name:50s} trees={m.best_iteration:5d}  median CV-RMSE={s['median_cv_rmse_pct']:6.2f}%  "
              f"|NMBE|={s['median_abs_nmbe_pct']:5.2f}%  ({time.time() - t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(REPORTS / "tuning_validation.csv", index=False)


if __name__ == "__main__":
    main()
