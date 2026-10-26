"""Feature engineering: calendar, weather, building profile and lagged consumption."""
import holidays
import numpy as np
import pandas as pd

from src.config import TRAIN_END

CATEGORICAL = ["building_id", "site", "primary_use"]
# dayofyear is left out on purpose: with only 18 months of history it lets the model memorise dates (see src/tune.py)
CALENDAR = ["hour", "dayofweek", "month", "hour_of_week", "is_weekend", "is_holiday"]
WEATHER = ["temperature", "humidity", "dew_point", "cooling_degree", "heating_degree",
           "temp_mean_24h", "temp_mean_72h", "temp_lag_3h", "temp_day_max", "temp_day_min"]
PROFILE = ["bld_mean_log", "bld_how_mean_log", "bld_month_mean_log", "bld_recent_mean_log", "bld_recent_how_log"]
LAGS = ["lag_24h", "lag_48h", "lag_168h", "lag_336h", "prev_day_mean", "prev_week_mean"]

# Weather model: what a building should use given weather and calendar (no recent meter data).
# Used for anomaly detection and savings verification (M&V).
WEATHER_MODEL_FEATURES = CATEGORICAL + CALENDAR + WEATHER + PROFILE
# Day-ahead model: also knows consumption up to the day before the forecast.
DAY_AHEAD_FEATURES = WEATHER_MODEL_FEATURES + LAGS

BASE_TEMP_C = 18.0
RECENT_DAYS = 56  # "recent level" = last 8 weeks before the end of training


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    us_holidays = holidays.country_holidays("US", years=sorted(ts.dt.year.unique()))
    df["hour"] = ts.dt.hour.astype(np.int8)
    df["dayofweek"] = ts.dt.dayofweek.astype(np.int8)
    df["month"] = ts.dt.month.astype(np.int8)
    df["dayofyear"] = ts.dt.dayofyear.astype(np.int16)
    df["hour_of_week"] = (df["dayofweek"].astype(np.int16) * 24 + df["hour"]).astype(np.int16)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_holiday"] = ts.dt.date.isin(set(us_holidays)).astype(np.int8)
    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    t, rh = df["temperature"], df["humidity"].clip(1, 100)
    gamma = np.log(rh / 100) + 17.62 * t / (243.12 + t)  # Magnus formula
    df["dew_point"] = (243.12 * gamma / (17.62 - gamma)).astype(np.float32)
    df["cooling_degree"] = (t - BASE_TEMP_C).clip(lower=0).astype(np.float32)
    df["heating_degree"] = (BASE_TEMP_C - t).clip(lower=0).astype(np.float32)

    # Weather is per site, so compute rolling features on the site series, then merge back
    site_w = df[["site", "timestamp", "temperature"]].drop_duplicates(["site", "timestamp"]).sort_values(["site", "timestamp"])
    g = site_w.groupby("site", observed=True)["temperature"]
    site_w["temp_mean_24h"] = g.transform(lambda s: s.rolling(24, min_periods=1).mean())
    site_w["temp_mean_72h"] = g.transform(lambda s: s.rolling(72, min_periods=1).mean())
    site_w["temp_lag_3h"] = g.shift(3).fillna(site_w["temperature"])
    day = site_w["timestamp"].dt.floor("D")
    site_w["temp_day_max"] = site_w.groupby(["site", day], observed=True)["temperature"].transform("max")
    site_w["temp_day_min"] = site_w.groupby(["site", day], observed=True)["temperature"].transform("min")
    site_w = site_w.drop(columns="temperature")
    for c in site_w.columns[2:]:
        site_w[c] = site_w[c].astype(np.float32)
    return df.merge(site_w, on=["site", "timestamp"], how="left")


def building_profiles(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Typical consumption of each building, learned from the training period only (no leakage)."""
    log_kwh = np.log1p(train["kwh"])
    t = train.assign(log_kwh=log_kwh)
    recent = t[t["timestamp"] >= t["timestamp"].max() - pd.Timedelta(days=RECENT_DAYS)]
    return {
        # Buildings drift over time (new equipment, occupancy); the latest level is the best guide to the future
        "bld_recent_mean_log": recent.groupby("building_id", observed=True)["log_kwh"].mean()
                                .rename("bld_recent_mean_log").reset_index(),
        "bld_recent_how_log": recent.groupby(["building_id", "hour_of_week"], observed=True)["log_kwh"].mean()
                               .rename("bld_recent_how_log").reset_index(),
        "bld_mean_log": t.groupby("building_id", observed=True)["log_kwh"].mean().rename("bld_mean_log").reset_index(),
        "bld_how_mean_log": t.groupby(["building_id", "hour_of_week"], observed=True)["log_kwh"].mean()
                             .rename("bld_how_mean_log").reset_index(),
        "bld_month_mean_log": t.groupby(["building_id", "month"], observed=True)["log_kwh"].mean()
                               .rename("bld_month_mean_log").reset_index(),
    }


def add_profiles(df: pd.DataFrame, profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.merge(profiles["bld_mean_log"], on="building_id", how="left")
    df = df.merge(profiles["bld_how_mean_log"], on=["building_id", "hour_of_week"], how="left")
    df = df.merge(profiles["bld_month_mean_log"], on=["building_id", "month"], how="left")
    df = df.merge(profiles["bld_recent_mean_log"], on="building_id", how="left")
    df = df.merge(profiles["bld_recent_how_log"], on=["building_id", "hour_of_week"], how="left")
    # A month never seen in training (e.g. a building with only 6 months of history) falls back to its mean
    df["bld_month_mean_log"] = df["bld_month_mean_log"].fillna(df["bld_mean_log"])
    df["bld_how_mean_log"] = df["bld_how_mean_log"].fillna(df["bld_mean_log"])
    df["bld_recent_mean_log"] = df["bld_recent_mean_log"].fillna(df["bld_mean_log"])
    df["bld_recent_how_log"] = df["bld_recent_how_log"].fillna(df["bld_how_mean_log"])
    for c in PROFILE:
        df[c] = df[c].astype(np.float32)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Past consumption known at forecast time (midnight of the day before): lags of 24h or more."""
    wide = df.pivot(index="timestamp", columns="building_id", values="kwh")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="h"))
    log_wide = np.log1p(wide)
    lag24 = log_wide.shift(24)
    lags = {
        "lag_24h": lag24,
        "lag_48h": log_wide.shift(48),
        "lag_168h": log_wide.shift(168),
        "lag_336h": log_wide.shift(336),
        "prev_day_mean": lag24.rolling(24, min_periods=12).mean(),
        "prev_week_mean": lag24.rolling(168, min_periods=48).mean(),
    }
    long = pd.concat(
        {name: frame.rename_axis("timestamp").stack(future_stack=True) for name, frame in lags.items()}, axis=1
    ).astype(np.float32).reset_index()
    long["building_id"] = long["building_id"].astype(df["building_id"].dtype)
    return df.merge(long, on=["timestamp", "building_id"], how="left")


def build_features(df: pd.DataFrame, train_end: str = TRAIN_END) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """train_end: profiles are learned only from data before this date (pass an earlier date for validation)."""
    df = add_calendar(df.copy())
    df = add_weather(df)
    profiles = building_profiles(df[df["timestamp"] < train_end])
    df = add_profiles(df, profiles)
    df = add_lags(df)
    df["split"] = np.where(df["timestamp"] < train_end, "train", "test")
    return df.sort_values(["building_id", "timestamp"]).reset_index(drop=True), profiles
