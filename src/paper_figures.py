"""Vector figures for the Baseline AI research paper (paper/figures/*.pdf)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, REPORTS, ROOT

OUT = ROOT / "paper" / "figures"
INK, GREY, ACCENT, SIGNAL, LIGHT = "#1b2830", "#8b9aa5", "#0a7a5c", "#c2410c", "#c9d3d0"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FEATURE_NAMES = {
    "bld_how_mean_log": "Building weekly profile", "bld_month_mean_log": "Building monthly profile",
    "building_id": "Building identity", "bld_mean_log": "Building mean level", "temperature": "Outdoor temperature",
    "bld_recent_how_log": "Recent weekly profile", "bld_recent_mean_log": "Recent mean level", "dew_point": "Dew point",
    "month": "Month", "hour_of_week": "Hour of week", "hour": "Hour of day", "lag_24h": "Load 24 h earlier",
    "lag_48h": "Load 48 h earlier", "lag_168h": "Load 1 week earlier", "lag_336h": "Load 2 weeks earlier",
    "prev_day_mean": "Previous-day mean", "prev_week_mean": "Previous-week mean", "humidity": "Humidity",
    "temp_mean_72h": "Temperature, 72 h mean", "temp_mean_24h": "Temperature, 24 h mean", "is_holiday": "Holiday",
    "dayofweek": "Day of week", "temp_day_max": "Daily max temperature", "temp_day_min": "Daily min temperature",
    "cooling_degree": "Cooling degree hours", "heating_degree": "Heating degree hours", "site": "Site",
    "primary_use": "Building type", "is_weekend": "Weekend", "temp_lag_3h": "Temperature 3 h earlier",
}


def model_comparison() -> None:
    res = pd.read_csv(REPORTS / "model_comparison.csv")
    res["label"] = res["model"].str.replace(" (weather model)", "").str.replace(" (day-ahead model)", "")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [5, 2]}, sharex=True)
    for ax, task, title in [(axes[0], "weather", "(a) Weather model task"), (axes[1], "day-ahead", "(b) Day-ahead task")]:
        r = res[res["task"] == task].sort_values("median_cv_rmse_pct", ascending=False)
        colors = [ACCENT if l.startswith("LightGBM") else GREY for l in r["label"]]
        bars = ax.barh(r["label"], r["median_cv_rmse_pct"], color=colors, height=0.62)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Median CV-RMSE (%)")
        ax.set_xlim(0, 33)
        ax.set_xticks([0, 10, 20, 30])
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.pdf")
    plt.close(fig)


def error_distribution() -> None:
    da = pd.read_csv(REPORTS / "per_building_lgbm_day_ahead.csv")["cv_rmse"]
    w = pd.read_csv(REPORTS / "per_building_lgbm_weather.csv")["cv_rmse"]
    bins = np.arange(0, 82, 2.5)
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    ax.hist(w.clip(upper=80), bins=bins, color=LIGHT, label=f"Weather model (median {w.median():.1f}%)")
    ax.hist(da.clip(upper=80), bins=bins, histtype="step", color=ACCENT, lw=1.4,
            label=f"Day-ahead model (median {da.median():.1f}%)")
    ax.axvline(30, color=INK, ls="--", lw=0.8)
    ax.text(30.8, ax.get_ylim()[1] * 0.92, "30% hourly limit", fontsize=7.5)
    ax.set_xlabel("CV-RMSE per building on the test period (%), values above 80% shown in the last bin")
    ax.set_ylabel("Buildings")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "error_distribution.pdf")
    plt.close(fig)


def example_weeks() -> None:
    pb = pd.read_csv(REPORTS / "per_building_lgbm_day_ahead.csv")
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet", filters=[("split", "==", "test")],
                           columns=["timestamp", "building_id", "kwh", "lgbm_weather", "lgbm_day_ahead"])
    pred["building_id"] = pred["building_id"].astype(str)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True)
    for ax, use in zip(axes, ["office", "education", "lodging"]):
        g = pb[pb["primary_use"] == use].sort_values("cv_rmse").reset_index(drop=True)
        row = g.iloc[len(g) // 2]  # the median building of that type
        b = pred[(pred["building_id"] == row["building_id"]) & (pred["timestamp"] >= "2017-10-02")
                 & (pred["timestamp"] < "2017-10-16")]
        ax.plot(b["timestamp"], b["kwh"], color=INK, lw=1.1, label="Metered")
        ax.plot(b["timestamp"], b["lgbm_weather"], color=GREY, lw=0.9, ls="--", label="Weather model")
        ax.plot(b["timestamp"], b["lgbm_day_ahead"], color=ACCENT, lw=1.0, label="Day-ahead model")
        ax.set_ylabel("kWh / h")
        ax.set_title(f"{use.capitalize()} building of median error (day-ahead CV-RMSE {row['cv_rmse']:.1f}%)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / "example_weeks.pdf")
    plt.close(fig)


def shap_importance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, name, title in [(axes[0], "lgbm_weather", "(a) Weather model"), (axes[1], "lgbm_day_ahead", "(b) Day-ahead model")]:
        imp = pd.read_csv(REPORTS / f"shap_importance_{name}.csv").head(10).iloc[::-1]
        ax.barh([FEATURE_NAMES.get(f, f) for f in imp["feature"]], imp["mean_abs_shap"],
                color=[ACCENT if i == len(imp) - 1 else GREY for i in range(len(imp))], height=0.62)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"Mean $|$SHAP$|$ (log-kWh)")
    fig.tight_layout()
    fig.savefig(OUT / "shap_importance.pdf")
    plt.close(fig)


def anomaly_example() -> None:
    daily = pd.read_parquet(DATA_PROCESSED / "daily_anomalies.parquet")
    daily["building_id"] = daily["building_id"].astype(str)
    events = pd.read_csv(REPORTS / "anomaly_events.csv")
    # a long, sizeable event in an ordinary building (moderate excess rather than a metering artefact)
    cand = events[events["days"].between(3, 7) & events["excess_pct"].between(25, 80)].sort_values("extra_kwh", ascending=False)
    bid = cand.iloc[0]["building_id"]
    d = daily[daily["building_id"] == bid].sort_values("date")
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    ax.bar(d["date"], d["actual"], color=np.where(d["anomaly"], SIGNAL, LIGHT), width=0.9, label="Metered (red: flagged)")
    ax.plot(d["date"], d["expected"], color=ACCENT, lw=1.1, label="Expected (calibrated weather model)")
    ax.set_ylabel("kWh / day")
    ax.set_title(f"Building {bid.split('_', 2)[-1]} ({bid.split('_')[1]}, site {bid.split('_')[0]})", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, max(d["actual"].max(), d["expected"].max()) * 1.25)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_example.pdf")
    plt.close(fig)
    return bid


def mv_errors() -> None:
    mv = pd.read_csv(REPORTS / "mv_per_building.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
    bins = np.arange(-40, 41, 2.5)
    axes[0].hist(mv["placebo_estimate_pct"].clip(-40, 40), bins=bins, color=LIGHT)
    axes[0].axvline(0, color=INK, ls="--", lw=0.8)
    axes[0].set_title("(a) Placebo: no saving applied", loc="left")
    axes[0].set_xlabel("Estimated saving (%)")
    axes[0].set_ylabel("Buildings")
    axes[1].hist(mv["saving_error_pct_points"].clip(-40, 40), bins=bins, color=ACCENT, alpha=0.85)
    axes[1].axvline(0, color=INK, ls="--", lw=0.8)
    axes[1].set_title("(b) Error on a true 15% saving", loc="left")
    axes[1].set_xlabel("Estimated minus true saving (percentage points)")
    fig.tight_layout()
    fig.savefig(OUT / "mv_errors.pdf")
    plt.close(fig)


def temperature_effect() -> None:
    import lightgbm as lgb

    from src.config import MODELS, SEED
    from src.features import WEATHER_MODEL_FEATURES
    x = pd.read_parquet(DATA_PROCESSED / "test_features.parquet", columns=WEATHER_MODEL_FEATURES)
    x = x.sample(20_000, random_state=SEED)
    model = lgb.Booster(model_file=str(MODELS / "lgbm_weather.txt"))
    sv = model.predict(x, pred_contrib=True)[:, :-1]
    t_idx, d_idx = WEATHER_MODEL_FEATURES.index("temperature"), WEATHER_MODEL_FEATURES.index("dew_point")
    weather_effect = sv[:, t_idx] + sv[:, d_idx]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.scatter(x["temperature"], weather_effect, s=1.5, alpha=0.25, color=ACCENT, rasterized=True)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("SHAP, temperature + dew point")
    fig.tight_layout()
    fig.savefig(OUT / "temperature_effect.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison()
    error_distribution()
    example_weeks()
    shap_importance()
    bid = anomaly_example()
    mv_errors()
    temperature_effect()
    print("Anomaly example building:", bid)
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
