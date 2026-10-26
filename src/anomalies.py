"""Anomaly detection and savings verification (M&V) on top of the weather model.

The weather model predicts what each building *should* use given weather and calendar.
- Anomaly: a day where actual use is far above that expectation.
- Savings (M&V): after a retrofit, savings = expected use - actual use.

Both are validated with synthetic experiments on real test data:
- inject +30% faults on known days and measure how many are detected
- inject a 15% retrofit saving and measure how well it is recovered
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS, REPORTS, SEED
from src.features import CATEGORICAL, WEATHER_MODEL_FEATURES, build_features
from src.train import LGB_PARAMS

PRED_COL = "lgbm_weather"
MIN_EXCESS = 0.15  # a day must be at least 15% above expected to be flagged
MAD_K = 2.5        # ... and an outlier for that building (median + 2.5 robust std)
CAL_DAYS = 28      # expected use is re-calibrated on the building's last 4 weeks ...
CAL_GAP = 7        # ... ending a week ago, so an ongoing fault cannot hide itself


def daily_totals(pred: pd.DataFrame) -> pd.DataFrame:
    d = pred.assign(date=pred["timestamp"].dt.floor("D"))
    daily = d.groupby(["building_id", "date"], observed=True).agg(
        actual=("kwh", "sum"), expected=(PRED_COL, "sum"), hours=("kwh", "size")).reset_index()
    daily = daily[daily["hours"] >= 20].sort_values(["building_id", "date"])
    # Buildings drift over months; a rolling calibration factor follows that drift
    g = daily.groupby("building_id", observed=True)
    past_actual = g["actual"].transform(lambda s: s.shift(CAL_GAP).rolling(CAL_DAYS, min_periods=14).sum())
    past_expected = g["expected"].transform(lambda s: s.shift(CAL_GAP).rolling(CAL_DAYS, min_periods=14).sum())
    daily["expected"] = daily["expected"] * (past_actual / past_expected).fillna(1.0)
    daily["excess"] = (daily["actual"] - daily["expected"]) / daily["expected"]
    return daily.reset_index(drop=True)


def flag_days(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("building_id", observed=True)["excess"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    daily = daily.assign(threshold=np.maximum(med + MAD_K * mad, MIN_EXCESS))
    daily["anomaly"] = daily["excess"] > daily["threshold"]
    return daily


def to_events(daily: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive anomalous days of one building into events."""
    a = daily[daily["anomaly"]].sort_values(["building_id", "date"])
    if a.empty:
        return pd.DataFrame()
    new_event = (a["building_id"] != a["building_id"].shift()) | (a["date"] - a["date"].shift() != pd.Timedelta("1D"))
    a = a.assign(event=new_event.cumsum())
    events = a.groupby("event").agg(
        building_id=("building_id", "first"), start=("date", "min"), end=("date", "max"), days=("date", "size"),
        actual_kwh=("actual", "sum"), expected_kwh=("expected", "sum"), max_excess=("excess", "max")).reset_index(drop=True)
    events["extra_kwh"] = events["actual_kwh"] - events["expected_kwh"]
    events["excess_pct"] = (events["extra_kwh"] / events["expected_kwh"] * 100).round(1)
    return events.sort_values("extra_kwh", ascending=False)


def validate_detection(test: pd.DataFrame, rng: np.random.Generator, size: float, n_buildings: int = 100) -> dict:
    """Inject a fault (e.g. +30%) for 5 days in random buildings and check how many fault days are caught."""
    buildings = rng.choice(test["building_id"].astype(str).unique(), n_buildings, replace=False)
    t = test.copy()
    dates = pd.date_range("2017-07-15", "2017-12-15", freq="D")
    injected = []
    for b in buildings:
        start = rng.choice(dates)
        mask = (t["building_id"] == b) & (t["timestamp"] >= start) & (t["timestamp"] < start + pd.Timedelta("5D"))
        t.loc[mask, "kwh"] *= 1 + size
        injected += [(b, start + pd.Timedelta(days=i)) for i in range(5)]
    daily = flag_days(daily_totals(t))
    inj = pd.DataFrame(injected, columns=["building_id", "date"]).assign(injected=True)
    inj["date"] = inj["date"].astype("datetime64[ns]")
    daily["building_id"] = daily["building_id"].astype(str)
    daily["date"] = daily["date"].astype("datetime64[ns]")
    m = daily.merge(inj, on=["building_id", "date"], how="left")
    m["injected"] = m["injected"].notna()
    sub = m[m["building_id"].isin(buildings)]
    assert sub["injected"].sum() > 0, "injected fault days did not match any daily rows"
    recall = sub.loc[sub["injected"], "anomaly"].mean()
    false_alarm = sub.loc[~sub["injected"], "anomaly"].mean()
    return {"buildings_tested": n_buildings, "fault_size": f"+{size:.0%} for 5 days",
            "fault_days_detected_pct": round(float(recall) * 100, 1),
            "normal_days_flagged_pct": round(float(false_alarm) * 100, 2)}


def validate_mv(true_saving: float = 0.15, retrofit_date: str = "2017-10-01") -> dict:
    """IPMVP-style test: train a baseline model on all data before the retrofit, apply a known saving
    to the metered data after it, and measure how well the saving is recovered."""
    df, _ = build_features(pd.read_parquet(DATA_PROCESSED / "dataset.parquet"), train_end=retrofit_date)
    pre, post = df[df["timestamp"] < retrofit_date], df[df["timestamp"] >= retrofit_date].copy()
    rounds = lgb.Booster(model_file=str(MODELS / "lgbm_weather.txt")).num_trees()
    model = lgb.train(LGB_PARAMS, lgb.Dataset(pre[WEATHER_MODEL_FEATURES], np.log1p(pre["kwh"]),
                                              categorical_feature=CATEGORICAL), num_boost_round=rounds)
    post["baseline"] = np.expm1(model.predict(post[WEATHER_MODEL_FEATURES]))
    totals = post.groupby("building_id", observed=True).agg(n=("kwh", "size"), actual=("kwh", "sum"),
                                                             baseline=("baseline", "sum"))
    totals = totals[totals["n"] >= 500]
    placebo = (1 - totals["actual"] / totals["baseline"]) * 100
    errors = (1 - totals["actual"] * (1 - true_saving) / totals["baseline"]) * 100 - true_saving * 100
    pd.DataFrame({"placebo_estimate_pct": placebo, "saving_error_pct_points": errors}).to_csv(
        REPORTS / "mv_per_building.csv")
    return {"buildings_tested": int(len(totals)), "true_saving_pct": true_saving * 100,
            "retrofit_date": retrofit_date, "measurement_months": 3,
            "median_abs_error_pct_points": round(float(errors.abs().median()), 2),
            "within_5_points_pct": round(float((errors.abs() <= 5).mean() * 100), 1),
            "within_10_points_pct": round(float((errors.abs() <= 10).mean() * 100), 1),
            "placebo_median_estimate_pct": round(float(placebo.median()), 2)}


def main() -> None:
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet",
                           columns=["timestamp", "building_id", "primary_use", "kwh", "split", PRED_COL])
    test = pred[pred["split"] == "test"].copy()
    rng = np.random.default_rng(SEED)

    daily = flag_days(daily_totals(test))
    daily.to_parquet(DATA_PROCESSED / "daily_anomalies.parquet", index=False)
    events = to_events(daily)
    events.to_csv(REPORTS / "anomaly_events.csv", index=False)
    print(f"Anomalies: {int(daily['anomaly'].sum())} anomalous building-days in {len(events)} events "
          f"({daily['anomaly'].mean() * 100:.1f}% of days)")
    if not events.empty:
        print(events.head(10).to_string(index=False))

    detection = validate_detection(test, rng, 0.30)
    print("Fault-injection test:", detection)
    detection_50 = validate_detection(test, np.random.default_rng(SEED), 0.50)
    print("Fault-injection test:", detection_50)
    mv = validate_mv()
    print("M&V test:", mv)
    (REPORTS / "validation.json").write_text(json.dumps(
        {"anomaly_detection": detection, "anomaly_detection_50": detection_50, "savings_verification": mv,
         "anomalous_day_share_pct": round(float(daily["anomaly"].mean() * 100), 2),
         "anomaly_events": int(len(events))}, indent=2))


if __name__ == "__main__":
    main()
