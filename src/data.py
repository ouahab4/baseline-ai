"""Load raw BDG2 files, clean the meter readings and build one long hourly table.

Output: data/processed/dataset.parquet with one row per building and hour:
timestamp, building_id, site, primary_use, kwh, temperature, humidity
"""
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, SITES, TRAIN_END, YEARS

MIN_TRAIN_HOURS = 2000  # a building needs enough history to learn from
MIN_TEST_HOURS = 1000   # ... and enough test data to be evaluated
STUCK_HOURS = 24        # identical readings this long = frozen meter
SPIKE_FACTOR = 5.0      # readings above 5x the building's 99th percentile = spike


def load_meters(site: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        wide = pd.read_csv(DATA_RAW / f"{site}_clean={year}.csv", parse_dates=["timestamp"], index_col="timestamp")
        if wide.shape[1] == 0:
            continue
        frames.append(wide.melt(ignore_index=False, var_name="building_id", value_name="kwh").reset_index())
    return pd.concat(frames, ignore_index=True)


def load_weather(site: str) -> pd.DataFrame:
    w = pd.read_csv(DATA_RAW / f"weather_{site}.csv", parse_dates=["timestamp"])
    w = w.groupby("timestamp").mean()
    full = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    w = w.reindex(full).interpolate(limit=6, limit_direction="both").rename_axis("timestamp").reset_index()
    w["site"] = site
    return w


def clean_building(s: pd.Series) -> pd.Series:
    """Mark impossible readings as missing: non-positive values, frozen meters and spikes."""
    s = s.where(s > 0)
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    s = s.where(run_len < STUCK_HOURS)
    q99 = s.quantile(0.99)
    return s.where(s <= SPIKE_FACTOR * q99)


def build_dataset() -> pd.DataFrame:
    meters = pd.concat([load_meters(s) for s in SITES], ignore_index=True)
    weather = pd.concat([load_weather(s) for s in SITES], ignore_index=True)

    parts = meters["building_id"].str.split("_", n=2, expand=True)
    meters["site"] = parts[0]
    meters["primary_use"] = parts[1]

    meters = meters.sort_values(["building_id", "timestamp"])
    raw_valid = meters["kwh"].notna().sum()
    meters["kwh"] = meters.groupby("building_id")["kwh"].transform(clean_building)
    print(f"  cleaning removed {raw_valid - meters['kwh'].notna().sum():,} suspicious readings")

    is_train = meters["timestamp"] < TRAIN_END
    counts = meters[meters["kwh"].notna()].assign(train=is_train).groupby(["building_id", "train"]).size().unstack(fill_value=0)
    keep = counts[(counts.get(True, 0) >= MIN_TRAIN_HOURS) & (counts.get(False, 0) >= MIN_TEST_HOURS)].index
    print(f"  kept {len(keep)} of {meters['building_id'].nunique()} buildings with enough train and test data")

    df = meters[meters["building_id"].isin(keep) & meters["kwh"].notna()]
    df = df.merge(weather, on=["site", "timestamp"], how="left")
    for col in ["building_id", "site", "primary_use"]:
        df[col] = df[col].astype("category")
    df["kwh"] = df["kwh"].astype(np.float32)
    df["temperature"] = df["temperature"].astype(np.float32)
    df["humidity"] = df["humidity"].astype(np.float32)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = build_dataset()
    out = DATA_PROCESSED / "dataset.parquet"
    data.to_parquet(out, index=False)
    print(f"  saved {len(data):,} rows, {data['building_id'].nunique()} buildings -> {out}")
