"""Import your own hourly meter data and prepare it for the pipeline.

Usage:
    python -m src.import_csv my_meters.csv --train-end YYYY-MM-DD

Required columns: timestamp, building_id, kwh, temperature, humidity
Optional columns: site (weather location, default "site"), primary_use (building type, default "other")

The file is cleaned with the same rules as the reference data and written to data/processed/dataset.parquet.
Then run: python -m src.train && python -m src.explain && python -m src.anomalies
"""
import argparse
import re

import numpy as np
import pandas as pd

from src import config
from src.data import MIN_TEST_HOURS, MIN_TRAIN_HOURS, clean_building

REQUIRED = ["timestamp", "building_id", "kwh", "temperature", "humidity"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="hourly meter file")
    ap.add_argument("--train-end", required=True, help="first day of the test period, format YYYY-MM-DD")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}. Required: {', '.join(REQUIRED)}")
    df["site"] = df["site"] if "site" in df.columns else "site"
    df["primary_use"] = df["primary_use"] if "primary_use" in df.columns else "other"
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.groupby(["building_id", "timestamp"], as_index=False).agg(
        kwh=("kwh", "sum"), temperature=("temperature", "mean"), humidity=("humidity", "mean"),
        site=("site", "first"), primary_use=("primary_use", "first"))

    df = df.sort_values(["building_id", "timestamp"])
    df["kwh"] = df.groupby("building_id")["kwh"].transform(clean_building)
    df = df.dropna(subset=["kwh", "temperature", "humidity"])
    is_train = df["timestamp"] < args.train_end
    counts = df.assign(train=is_train).groupby(["building_id", "train"]).size().unstack(fill_value=0)
    keep = counts[(counts.get(True, 0) >= MIN_TRAIN_HOURS) & (counts.get(False, 0) >= MIN_TEST_HOURS)].index
    if len(keep) == 0:
        raise SystemExit(f"No building has {MIN_TRAIN_HOURS} training hours and {MIN_TEST_HOURS} test hours.")
    df = df[df["building_id"].isin(keep)]
    for col in ["building_id", "site", "primary_use"]:
        df[col] = df[col].astype(str).astype("category")
    for col in ["kwh", "temperature", "humidity"]:
        df[col] = df[col].astype(np.float32)

    out = config.DATA_PROCESSED / "dataset.parquet"
    df.reset_index(drop=True).to_parquet(out, index=False)

    # Point the pipeline at the new train/test boundary
    cfg = config.ROOT / "src" / "config.py"
    text = re.sub(r'TRAIN_END = "[^"]+"', f'TRAIN_END = "{args.train_end}"', cfg.read_text())
    cfg.write_text(text)
    print(f"Imported {len(df):,} rows for {df['building_id'].nunique()} buildings -> {out}")
    print(f"Test period starts {args.train_end}. Next: python -m src.train")


if __name__ == "__main__":
    main()
