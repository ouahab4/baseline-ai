"""Explainability (SHAP) and report figures."""
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import DATA_PROCESSED, FIGURES, MODELS, REPORTS, SEED
from src.features import DAY_AHEAD_FEATURES, WEATHER_MODEL_FEATURES, build_features

GREY, ACCENT = "#9aa0a6", "#1f6fd1"


def shap_values(model: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values computed by LightGBM itself (last column = base value)."""
    return model.predict(x, pred_contrib=True)[:, :-1]


def plot_model_comparison() -> None:
    res = pd.read_csv(REPORTS / "model_comparison.csv").sort_values("median_cv_rmse_pct")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [ACCENT if "LightGBM" in m else GREY for m in res["model"]]
    bars = ax.barh(res["model"] + "  [" + res["task"] + "]", res["median_cv_rmse_pct"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("Median CV-RMSE across buildings, test period (lower is better)")
    ax.set_title("LightGBM gives the lowest error on unseen months")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_example_weeks(pred: pd.DataFrame) -> None:
    test = pred[pred["split"] == "test"]
    examples = ["office", "education", "lodging"]
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 8), sharex=True)
    week = (test["timestamp"] >= "2017-10-02") & (test["timestamp"] < "2017-10-16")
    for ax, use in zip(axes, examples):
        bids = test.loc[week & (test["primary_use"] == use), "building_id"].astype(str).unique()
        bid = sorted(bids)[0]
        g = test[week & (test["building_id"] == bid)]
        ax.plot(g["timestamp"], g["kwh"], color="black", lw=1.4, label="Actual")
        ax.plot(g["timestamp"], g["lgbm_weather"], color=GREY, lw=1.2, label="Weather model")
        ax.plot(g["timestamp"], g["lgbm_day_ahead"], color=ACCENT, lw=1.2, label="Day-ahead model")
        ax.set_title(f"{bid} ({use})", fontsize=10, loc="left")
        ax.set_ylabel("kWh")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, loc="upper right", fontsize=8)
    fig.suptitle("Two weeks of unseen data: predictions follow daily and weekly cycles", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "example_weeks.png", dpi=150)
    plt.close(fig)


def plot_error_distribution() -> None:
    pb = pd.read_csv(REPORTS / "per_building_lgbm_weather.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pb["cv_rmse"].clip(upper=80), bins=40, color=GREY)
    med = pb["cv_rmse"].median()
    ax.axvline(30, color="black", ls="--", lw=1)
    ax.text(30.5, ax.get_ylim()[1] * 0.9, "ASHRAE G14 limit (30%)", fontsize=8)
    ax.axvline(med, color=ACCENT, lw=2)
    ax.text(med + 0.5, ax.get_ylim()[1] * 0.75, f"median {med:.1f}%", color=ACCENT, fontsize=9)
    ax.set_xlabel("CV-RMSE per building, weather model, test period (%) — values above 80% shown at 80")
    ax.set_ylabel("Buildings")
    ax.set_title("Most buildings are predicted within the ASHRAE Guideline 14 limit", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_distribution.png", dpi=150)
    plt.close(fig)


def explain_model(name: str, features: list[str], test: pd.DataFrame) -> pd.DataFrame:
    model = lgb.Booster(model_file=str(MODELS / f"{name}.txt"))
    x = test[features].sample(20_000, random_state=SEED)
    sv = shap_values(model, x)
    importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(REPORTS / f"shap_importance_{name}.csv", index=False)

    x_plot = x.copy()
    for c in x_plot.select_dtypes("category").columns:
        x_plot[c] = x_plot[c].cat.codes
    plt.figure()
    shap.summary_plot(sv, x_plot, max_display=15, show=False)
    plt.title(f"What drives the {name.replace('lgbm_', '').replace('_', '-')} model (SHAP, log-kWh)", loc="left")
    plt.tight_layout()
    plt.savefig(FIGURES / f"shap_summary_{name}.png", dpi=150)
    plt.close()

    # Temperature effect for the weather model: how SHAP of temperature changes with temperature
    if name == "lgbm_weather":
        idx = features.index("temperature")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x["temperature"], sv[:, idx], s=2, alpha=0.3, color=ACCENT)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Outdoor temperature (°C)")
        ax.set_ylabel("SHAP value of temperature (log-kWh)")
        ax.set_title("Hot hours push consumption up: the cooling signature", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / "shap_temperature_dependence.png", dpi=150)
        plt.close(fig)
    return importance


def main() -> None:
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    print("Figures ...")
    plot_model_comparison()
    plot_example_weeks(pred)
    plot_error_distribution()

    print("Rebuilding features for SHAP and the dashboard ...")
    df, _ = build_features(pd.read_parquet(DATA_PROCESSED / "dataset.parquet"))
    test = df[df["split"] == "test"]
    test[["timestamp"] + DAY_AHEAD_FEATURES].to_parquet(DATA_PROCESSED / "test_features.parquet", index=False)

    for name, feats in [("lgbm_weather", WEATHER_MODEL_FEATURES), ("lgbm_day_ahead", DAY_AHEAD_FEATURES)]:
        imp = explain_model(name, feats, test)
        print(f"  {name} top drivers: " + ", ".join(imp["feature"].head(6)))
    print("Saved figures to", FIGURES)


if __name__ == "__main__":
    main()
