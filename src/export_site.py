"""Export real model results to docs/data/results.json for the website's interactive platform.

Buildings are chosen to be representative, not flattering: for each common building type we take the
buildings at the 25th, 50th and 75th percentile of day-ahead error.
"""
import json

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, REPORTS, ROOT, TRAIN_END

SITE_DATA = ROOT / "docs" / "data"
TYPES = ["education", "office", "lodging", "public", "assembly", "health", "retail", "warehouse"]


def rounded(values: np.ndarray) -> list:
    return [None if np.isnan(v) else round(float(v), 2) for v in values]


def pick_buildings() -> list[str]:
    pb = pd.read_csv(REPORTS / "per_building_lgbm_day_ahead.csv")
    chosen = []
    for use in TYPES:
        g = pb[pb["primary_use"] == use].sort_values("cv_rmse").reset_index(drop=True)
        if g.empty:
            continue
        for q in (0.25, 0.5, 0.75):
            bid = g.loc[int(round(q * (len(g) - 1))), "building_id"]
            if bid not in chosen:
                chosen.append(bid)
    return chosen


def main() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet", filters=[("split", "==", "test")],
                           columns=["timestamp", "building_id", "site", "primary_use", "kwh", "temperature",
                                    "lgbm_weather", "lgbm_day_ahead"])
    pred["building_id"] = pred["building_id"].astype(str)
    daily = pd.read_parquet(DATA_PROCESSED / "daily_anomalies.parquet")
    daily["building_id"] = daily["building_id"].astype(str)
    hours = pd.date_range(TRAIN_END, "2017-12-31 23:00", freq="h")

    buildings = []
    for bid in pick_buildings():
        g = pred[pred["building_id"] == bid].set_index("timestamp").reindex(hours)
        d = daily[daily["building_id"] == bid].sort_values("date")
        buildings.append({
            "id": bid,
            "site": bid.split("_")[0],
            "use": bid.split("_")[1],
            "actual": rounded(g["kwh"].to_numpy()),
            "weather_model": rounded(g["lgbm_weather"].to_numpy()),
            "day_ahead": rounded(g["lgbm_day_ahead"].to_numpy()),
            "temperature": rounded(g["temperature"].to_numpy()),
            "daily": {
                "date": d["date"].dt.strftime("%Y-%m-%d").tolist(),
                "actual": rounded(d["actual"].to_numpy()),
                "expected": rounded(d["expected"].to_numpy()),
                "anomaly": d["anomaly"].astype(int).tolist(),
            },
        })

    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    validation = json.loads((REPORTS / "validation.json").read_text())
    per_bld = pd.read_csv(REPORTS / "per_building_lgbm_day_ahead.csv")
    by_use = pd.read_csv(REPORTS / "by_primary_use_lgbm_day_ahead.csv")
    shap_w = pd.read_csv(REPORTS / "shap_importance_lgbm_weather.csv").head(10)
    shap_d = pd.read_csv(REPORTS / "shap_importance_lgbm_day_ahead.csv").head(10)
    data = pd.read_parquet(DATA_PROCESSED / "dataset.parquet", columns=["building_id", "site"])

    out = {
        "start": str(hours[0]),
        "portfolio": {"buildings": int(data["building_id"].nunique()), "sites": int(data["site"].nunique()),
                      "hourly_readings": int(len(data)), "train_period": "2016-01-01 to 2017-06-30",
                      "test_period": "2017-07-01 to 2017-12-31"},
        "comparison": comparison.drop(columns="column").to_dict(orient="records"),
        "validation": validation,
        "cv_rmse_day_ahead": rounded(per_bld["cv_rmse"].to_numpy()),
        "by_use": by_use.to_dict(orient="records"),
        "shap_weather": shap_w.round(4).to_dict(orient="records"),
        "shap_day_ahead": shap_d.round(4).to_dict(orient="records"),
        "buildings": buildings,
    }
    path = SITE_DATA / "results.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Exported {len(buildings)} buildings -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
