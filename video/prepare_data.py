"""Real data for the film, taken from the website export (docs/data/results.json)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILDING = "Fox_office_Zachary"
HOURS_PER_DAY = 24


def clean(values):
    return [None if v is None else round(float(v), 1) for v in values]


def main() -> None:
    data = json.loads((ROOT / "docs" / "data" / "results.json").read_text())
    b = next(x for x in data["buildings"] if x["id"] == BUILDING)
    actual = np.array([np.nan if v is None else v for v in b["actual"]], dtype=float)
    weather = np.array([np.nan if v is None else v for v in b["weather_model"]], dtype=float)

    # One month of hours (Oct) for the "monthly bill" scene, one week (from 9 Oct) for the forecast scene
    month = slice(92 * HOURS_PER_DAY, 123 * HOURS_PER_DAY)
    week = slice(100 * HOURS_PER_DAY, 107 * HOURS_PER_DAY)

    # Savings illustration: daily totals, model calibrated on the 8 weeks before the upgrade, 15% saving applied after
    days = len(actual) // HOURS_PER_DAY
    da = np.nansum(actual[: days * 24].reshape(days, 24), axis=1)
    dw = np.nansum(weather[: days * 24].reshape(days, 24), axis=1)
    cut = 92
    cal = da[cut - 56:cut].sum() / dw[cut - 56:cut].sum()
    after = da.copy()
    after[cut:] *= 0.85

    out = {
        "building": "Office building",
        "month_actual": clean(actual[month]),
        "month_expected": clean(weather[month]),
        "month_total_kwh": int(np.nansum(actual[month])),
        "week_actual": clean(actual[week]),
        "week_forecast": clean(b["day_ahead"][week]),
        "daily_actual": clean(b["daily"]["actual"]),
        "daily_expected": clean(b["daily"]["expected"]),
        "daily_anomaly": b["daily"]["anomaly"],
        "save_baseline": clean(dw * cal),
        "save_actual": clean(after),
        "save_cut": cut,
        "buildings": data["portfolio"]["buildings"],
    }
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "build" / "film_data.json").write_text(json.dumps(out))
    print("month total kWh:", out["month_total_kwh"], "| anomaly days:", sum(out["daily_anomaly"]))


if __name__ == "__main__":
    main()
