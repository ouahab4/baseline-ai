"""Energy forecasting metrics (ASHRAE Guideline 14 style)."""
import numpy as np
import pandas as pd


def cv_rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of variation of the RMSE, in percent of mean consumption."""
    return float(np.sqrt(np.mean((y - p) ** 2)) / np.mean(y) * 100)


def nmbe(y: np.ndarray, p: np.ndarray) -> float:
    """Normalized mean bias error, in percent. Positive = model under-predicts."""
    return float(np.sum(y - p) / (len(y) * np.mean(y)) * 100)


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(p, 0, None))) ** 2)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p) / y) * 100)


def per_building(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for bid, g in df.groupby("building_id", observed=True):
        y, p = g["kwh"].to_numpy(), g[pred_col].to_numpy()
        rows.append({"building_id": bid, "primary_use": g["primary_use"].iloc[0],
                     "cv_rmse": cv_rmse(y, p), "nmbe": nmbe(y, p), "mape": mape(y, p)})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Headline numbers for one model on one dataset."""
    d = df.dropna(subset=[pred_col])
    y, p = d["kwh"].to_numpy(), d[pred_col].to_numpy()
    pb = per_building(d, pred_col)
    passes_g14 = (pb["cv_rmse"] <= 30) & (pb["nmbe"].abs() <= 10)
    return {
        "median_cv_rmse_pct": round(float(pb["cv_rmse"].median()), 2),
        "median_abs_nmbe_pct": round(float(pb["nmbe"].abs().median()), 2),
        "median_mape_pct": round(float(pb["mape"].median()), 2),
        "rmsle": round(rmsle(y, p), 4),
        "mae_kwh": round(float(np.mean(np.abs(y - p))), 2),
        "ashrae_g14_pass_pct": round(float(passes_g14.mean() * 100), 1),
        "n_buildings": int(len(pb)),
    }
